import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import os

from tokenizer import SimpleTokenizer
from dataset import SpeechesClassificationDataset, LanguageModelingDataset

from transformer import SpeechClassifier, Decoder
from utilities import Utilities

seed = 42

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Hyperparameters to use for training to roughly match 
the numbers mentioned in the assignment description """
batch_size = 16  # Number of independent sequences  we will process in parallel
block_size = 32  # Maximum context length for predictions
learning_rate = 1e-3  # Learning rate for the optimizer
n_embd = 64  # Embedding dimension
n_head = 2  # Number of attention heads
n_layer = 4  # Number of transformer layers


eval_interval = 100  # How often to evaluate train and test perplexity during training
max_iters = 500 # For language modeling, we can process all the batches for the entire dataset, but that takes a while, so we'll limit it to 500 iterations. For batch size of 16 and block size of  32, this is roughly, this is  500 * 16 * 32 = 256000 tokens, SOTA LMs are trained on trillions of tokens, so this is a very small dataset.
eval_iters = 200  # Number of iterations to evaluate perplexity on the test set


## classifier training hyperparameters. It is a simple 1 hidden layer feedforward network, with input 
## size of 64, hidden size of 100 and output size of 3.

n_input = 64  # Input size for the classifier, should match the embedding size of the transformer
n_hidden = 100  # Hidden size for the classifier
n_output = 3  # Output size for the classifier, we have 3 classes
epochs_CLS = 15 # epochs for classifier training

def load_texts(directory):
    """
    This function loads all texts from the specified directory, ignoring any files with "test" in their name. The text is used for "training" the tokenizer. Since our tokenizer is simple, we don't need to do any training, but we still need to ignore the test data. 
    """

    texts = []
    files = os.listdir(directory)
    for filename in files: 
        if "test" in filename:  ## don't "read test files"
            continue
        with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
    return texts



def collate_batch(batch):
    """ Collate a batch of data into a single tensor with padding."""
    data, labels = zip(*batch)  # Separate the data and labels
    # Pad sequences to the fixed length
    padded_sequences = pad_sequence(data, batch_first=True, padding_value=0)
    padded_sequences = padded_sequences[:, :block_size]  # Truncate if longer
    # Add padding if shorter
    padded_sequences = torch.nn.functional.pad(padded_sequences, (0, max(0, block_size - padded_sequences.shape[1])), "constant", 0)
    labels = torch.stack(labels)  
    return padded_sequences, labels

def compute_classifier_accuracy(classifier, data_loader):
    """ Compute the accuracy of the classifier on the data in data_loader."""
    classifier.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for X, Y in data_loader:
            X, Y = X.to(device), Y.to(device)
            outputs = classifier(X)
            _, predicted = torch.max(outputs.data, 1)
            total_correct += (predicted == Y).sum().item()
            total_samples += Y.size(0)
        accuracy = (100 * total_correct / total_samples)
        classifier.train()
        return accuracy


def compute_perplexity(decoderLMmodel, data_loader, eval_iters=100):
    """ Compute the perplexity of the decoderLMmodel on the data in data_loader.
    Make sure to use the cross entropy loss for the decoderLMmodel.
    """
    decoderLMmodel.eval()
    losses= []
    total_loss = 0
    for X, Y in data_loader:
        X, Y = X.to(device), Y.to(device)
        loss = decoderLMmodel(X, Y) # your model should be computing the cross entropy loss
        losses.append(loss.item())
        total_loss += loss.item()
        if len(losses) >= eval_iters: break

    print("total loss: ", total_loss)
    losses = torch.tensor(losses)
    mean_loss = losses.mean()
    perplexity = torch.exp(mean_loss).item()  # Calculate perplexity as exp(mean loss)

    decoderLMmodel.train()
    return perplexity

def main():

    print("Loading data and creating tokenizer ...")
    texts = load_texts('speechesdataset')
    tokenizer = SimpleTokenizer(' '.join(texts)) # create a tokenizer from the data
    print("Vocabulary size is", tokenizer.vocab_size)

    train_CLS_dataset = SpeechesClassificationDataset(
        tokenizer,"speechesdataset/train_CLS.tsv")
    train_CLS_loader = DataLoader(train_CLS_dataset,
                                  batch_size=batch_size,
                                  collate_fn=collate_batch,
                                  shuffle=True)

  
    input_file = "speechesdataset/train_LM.txt"
    with open(input_file, 'r', encoding='utf-8') as f:
        lmtrainText = f.read()
    train_LM_dataset = LanguageModelingDataset(tokenizer, lmtrainText,  block_size)
    train_LM_loader = DataLoader(train_LM_dataset, batch_size=batch_size, shuffle=True)

    test_LM_files = ["test_LM_hbush.txt", "test_LM_obama.txt", "test_LM_wbush.txt"]
    test_LM_loaders = []

    for filename in test_LM_files:
        with open("speechesdataset/" + filename, "r", encoding="utf-8") as file:
            text = file.read()
        dataset = LanguageModelingDataset(tokenizer, text, block_size)
        loader = DataLoader(dataset,
                            batch_size,
                            shuffle=False)
        test_LM_loaders.append(loader)

    test_CLS_dataset = SpeechesClassificationDataset(
        tokenizer,"speechesdataset/test_CLS.tsv")
    test_CLS_loader = DataLoader(test_CLS_dataset,
                                 batch_size=batch_size,
                                 collate_fn=collate_batch,
                                 shuffle=False)

    model = SpeechClassifier(vocab_size=tokenizer.vocab_size,
                             block_size=block_size,
                             n_embed=n_embd,
                             n_head=n_head,
                             n_layer=n_layer,
                             n_hidden=n_hidden,
                             n_output=n_output).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()
    encoder_parameters = sum(p.nelement() for p in model.encoder.parameters())
    print("encoder parameters: ", encoder_parameters)

    # CLS training code here
    for epoch in range(epochs_CLS):
        model.train()
        total_loss = 0.0
        for xb, yb in train_CLS_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            scores, attn_matrices = model(xb)
            loss = loss_fn(scores,yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for xb, yb in test_CLS_loader:
                xb, yb = xb.to(device), yb.to(device)
                scores, _ = model(xb)
                predictions = scores.argmax(dim=1)
                correct = correct + (predictions==yb).sum().item()
                total = total + yb.size(0)

        accuracy = (100 * correct) / total
        average_loss = total_loss / len(train_CLS_loader)

        print("Epoch ", {(epoch+1)/epochs_CLS})
        print("loss: ", average_loss)
        print("test accuracy", accuracy)

    utils = Utilities(tokenizer, model)
    utils.sanity_check("We must work together to build a better future", block_size)

    decoder = Decoder(vocab_size=tokenizer.vocab_size,
                      block_size=block_size,
                      n_embed=n_embd,
                      n_head=n_head,
                      n_layer=n_layer).to(device)
    optimizer = torch.optim.Adam(decoder.parameters(),
                                 lr = learning_rate)
    decoder.train()
    decoder_parameters = sum(param.numel() for param in decoder.parameters()
                             if param.requires_grad)
    print("Decoder Parameters: ", decoder_parameters)
    # for the language modeling task, you will iterate over the training data for a fixed number of iterations like this:
    for i, (xb, yb) in enumerate(train_LM_loader):
        if i >= max_iters:
            break
        xb, yb = xb.to(device), yb.to(device)

        #get predictions + calculate loss
        loss = decoder(xb, yb)
        #update the weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % eval_interval == 0:
            print("iteration: ", i)

    print("Total training batches: ", len(train_LM_loader))
    train_perplexity = compute_perplexity(decoder, train_LM_loader, eval_iters)
    hbush_perplexity = compute_perplexity(decoder, test_LM_loaders[0], eval_iters)
    obama_perplexity = compute_perplexity(decoder, test_LM_loaders[1], eval_iters)
    wbush_perplexity = compute_perplexity(decoder, test_LM_loaders[2], eval_iters)

    print("Train Perplexity: ", train_perplexity)
    print("HBush Perplexity: ", hbush_perplexity)
    print("Obama Perplexity: ", obama_perplexity)
    print("WBush Perplexity: ", wbush_perplexity)
    utils = Utilities(tokenizer, decoder)
    utils.sanity_check("We must work together to build a better future", block_size)


if __name__ == "__main__":
    main()
