import torch
import torch.nn as nn
import torch.nn.functional as F


# add all  your Encoder and Decoder code here

class FeedForward(nn.Module):
    def __init__(self, n_embed, hidden_size):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(n_embed, hidden_size),
                                     nn.ReLU(),
                                     nn.Dropout(0.1),
                                     nn.Linear(hidden_size, n_embed))

    def forward(self, x):
        return self.network(x)

class MultiheadAttention(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()

        assert n_embed % n_head == 0

        self.n_head = n_head
        self.head_size = n_embed // n_head

        self.query = nn.Linear(n_embed, n_embed)
        self.key = nn.Linear(n_embed, n_embed)
        self.value = nn.Linear(n_embed, n_embed)
        self.output_projection = nn.Linear(n_embed, n_embed)

    def forward(self, x, padding_mask=None, causal=False):
        batch_size, sequence_len, embed_size = x.shape

        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        #turn the vectors into attention heads
        query = query.reshape(batch_size,
                              sequence_len,
                              self.n_head,
                              self.head_size
                              ).transpose(1,2)
        key = key.reshape(batch_size,
                              sequence_len,
                              self.n_head,
                              self.head_size
                              ).transpose(1,2)
        value = value.reshape(batch_size,
                              sequence_len,
                              self.n_head,
                              self.head_size
                              ).transpose(1,2)

        #find the scaled dot product attention score
        scores = torch.matmul(query, key.transpose(-2,-1))
        scores = scores / (self.head_size**0.5)

        if padding_mask is not None:
            is_padding = padding_mask[:, None, None, :] == False
            scores = scores.masked_fill(is_padding, float("-inf"))

        if causal:
            future_mask = torch.triu(torch.ones(sequence_len,
                                                sequence_len,
                                                device=x.device,
                                                dtype=torch.bool),
                                     diagonal=1)
            scores = scores.masked_fill(future_mask[None,None,:,:], float("-inf"))

        attention_weights = F.softmax(scores, dim=-1)

        #use attention weights to combine value vectors
        out = torch.matmul(attention_weights, value)

        #combine attention heads
        out = out.transpose(1,2).contiguous()
        out = out.reshape(batch_size, sequence_len, embed_size)
        out = self.output_projection(out)

        return out, attention_weights

class TransformerBlock(nn.Module):
    def __init__(self, n_embed, n_head, hidden_size):
        super().__init__()
        self.attention = MultiheadAttention(n_embed,n_head)
        self.ff = FeedForward(n_embed, hidden_size)

        self.norm1 = nn.LayerNorm(n_embed)
        self.norm2 = nn.LayerNorm(n_embed)

    def forward(self, x, padding_mask = None, causal = False):
        #create attention layer w/ a residual connection
        attn_output, attn_weights = self.attention(self.norm1(x),
                                                   padding_mask=padding_mask,
                                                   causal=causal)
        x = x + attn_output

        #feed-forward network, add another residual connection
        x = x + self.ff(self.norm2(x))

        return x, attn_weights

class Encoder(nn.Module):
    def __init__(self, vocab_size, block_size, n_embed=64, n_head=2, n_layer=4):
        super().__init__()

        self.block_size = block_size

        #get the learned token and position embeddings
        self.token_embeddings = nn.Embedding(vocab_size, n_embed)
        self.position_embeddings = nn.Embedding(block_size, n_embed)

        #stack transformer blocks to use in the forward method
        self.blocks = nn.ModuleList([TransformerBlock(n_embed, n_head, 4 * n_embed)
                                     for _ in range(n_layer)])

        self.final_normalization = nn.LayerNorm(n_embed)

    def forward(self, x):
        batch_size, sequence_len = x.shape
        if sequence_len > self.block_size:
            raise IndexError

        #if the token id is 0 we know it's padding so remove it from our set
        padding_mask = (x!=0)

        positions = torch.arange(sequence_len, device=x.device)

        x = self.token_embeddings(x) + self.position_embeddings(positions).unsqueeze(0)

        attn_matrices = []

        for block in self.blocks:
            x, attn_weights = block(x, padding_mask=padding_mask)
            attn_matrices.append(attn_weights)

        x = self.final_normalization(x)

        return x, attn_matrices

class FeedForwardClassifier(nn.Module):
    def __init__(self, n_embed = 64, n_hidden = 100, n_output = 3):
        super().__init__()

        self.network = nn.Sequential(nn.Linear(n_embed,n_hidden),
                                     nn.ReLU(),
                                     nn.Linear(n_hidden, n_output))

    def forward(self, x):
        return self.network(x)

class SpeechClassifier(nn.Module):
    def __init__(self, vocab_size, block_size, n_embed=64, n_head=2,
                 n_layer=4, n_hidden=100, n_output=3):
        super().__init__()

        self.encoder = Encoder(vocab_size, block_size, n_embed, n_head, n_layer)
        self.classifier = FeedForwardClassifier(n_embed, n_hidden, n_output)

    def forward(self, x):
        encoded, attn_matrices = self.encoder(x)
        #ignore padding
        mask = (x != 0)
        mask = mask.reshape(x.shape[0], x.shape[1], 1)
        mask = mask.to(encoded.dtype)

        summed_vectors = (encoded * mask).sum(dim=1)
        token_counts = mask.sum(dim=1)
        if (token_counts == 0).any():
            token_counts[token_counts==0] = 1

        sentence_vectors = summed_vectors / token_counts
        scores = self.classifier(sentence_vectors)
        return scores, attn_matrices

class Decoder(nn.Module):
    def __init__(self, vocab_size, block_size, n_embed=64, n_head=2, n_layer=4):
        super().__init__()

        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embed)
        self.position_embedding = nn.Embedding(block_size, n_embed)
        self.blocks = nn.ModuleList([TransformerBlock(n_embed,n_head, 100)
                                     for _ in range(n_layer)])
        self.final_norm = nn.LayerNorm(n_embed)
        self.output_proj = nn.Linear(n_embed, vocab_size)

    def forward(self, x, targets = None):
        batch_size, sequence_len = x.shape
        if sequence_len > self.block_size:
            raise IndexError

        positions = torch.arange(sequence_len, device=x.device)
        x = self.token_embedding(x) + self.position_embedding(positions).unsqueeze(0)

        attn_matrices = []
        for block in self.blocks:
            x, attn_weights = block(x, causal = True)
            attn_matrices.append(attn_weights)

        x = self.final_norm(x)
        scores = self.output_proj(x)

        if targets is not None:
            loss = F.cross_entropy(
                scores.reshape(-1, scores.size(-1)),
                targets.reshape(-1)
            )
            return loss

        return scores, attn_matrices






