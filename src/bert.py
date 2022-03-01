# Use BERT to categorize actions as helpful, neutral, or harmful for the environment

import torch, gc
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import *
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import Adam
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from tqdm import tqdm, trange

#---------------------GPU setup---------------------#
cuda_num = 1

device = torch.device("cuda:{}".format(cuda_num) if torch.cuda.is_available() else "cpu")
print('Device: ', device)

n_gpu = torch.cuda.device_count()

#------------------Free up memory-------------------#
gc.collect()
torch.cuda.empty_cache()

categories = ['negative', 'neutral', 'positive']

model_class = BertForSequenceClassification
tokenizer_class = BertTokenizer
pretrained_weights = 'bert-base-uncased'

#----------Load in the model and tokenizer----------#
print('- - - Loading BERT model - - -')
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights, num_labels=2)



# Import a dataset
print('- - - Reading in data - - - ')
train_df = pd.read_csv('data/train_en.tsv', header=0, sep='\t')
dev_df = pd.read_csv('data/dev_en.tsv', header=0, sep='\t')
test_df = pd.read_csv('data/dev_en.tsv', header=0, sep='\t')

print(train_df)
print(dev_df)
print(test_df)



# Extract tweets and labels
print('- - - Preprocessing training data - - -')
train_tweets = []
train_labels = []

for i in range(len(train_df)):
    tweet = train_df.iloc[i]['text'][:512]            # Text (BERT only takes up to 512)
    label = train_df.iloc[i]['HS']                    # Label (0, 1)
    tweet_tokens = tokenizer.encode(tweet)      # Tokenize text
    padding = [0] * (512 - len(tweet_tokens))   # Add padding to text
    tweet_tokens += padding
    train_tweets.append(tweet_tokens)           # Tensorize each tweet
    train_labels.append(label)

# Validate train data quality
print(train_tweets[0])
print(train_labels[0])



print('- - - Preprocessing development data - - -')
dev_tweets = []
dev_labels = []

for i in range(len(dev_df)):
    tweet = dev_df.iloc[i]['text'][:512]            # Text (BERT only takes up to 512)
    label = dev_df.iloc[i]['HS']                    # Label (0, 1)
    tweet_tokens = tokenizer.encode(tweet)      # Tokenize text
    padding = [0] * (512 - len(tweet_tokens))   # Add padding to text
    tweet_tokens += padding
    dev_tweets.append(tweet_tokens)             # Tensorize each tweet
    dev_labels.append(label)

# Validate dev data quality
print(dev_tweets[0])
print(dev_labels[0])



print('- - - Preprocessing test data - - -')
test_tweets = []
test_labels = []


for i in range(len(test_df)):
    tweet = test_df.iloc[i]['text'][:512]            # Text (BERT only takes up to 512)
    label = test_df.iloc[i]['HS']                    # Label (0, 1)
    tweet_tokens = tokenizer.encode(tweet)      # Tokenize text
    padding = [0] * (512 - len(tweet_tokens))   # Add padding to text
    tweet_tokens += padding
    test_tweets.append(tweet_tokens)           # Tensorize each tweet
    test_labels.append(label)

# Validate test data quality
print(test_tweets[0])
print(test_labels[0])



# Tensorize
train_tweets = torch.tensor(train_tweets)
train_labels = torch.tensor(train_labels)
dev_tweets = torch.tensor(dev_tweets)
dev_labels = torch.tensor(dev_labels)
test_tweets = torch.tensor(test_tweets)
test_labels = torch.tensor(test_labels)



# Create dataset, sampler, and loader
epochs = 10
batch_size = 16
learning_rate = 2e-5
adam_epsilon = 1e-8
weight_decay = 0.0
max_grad_norm = 1.0

train_data = TensorDataset(train_tweets, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

dev_data = TensorDataset(dev_tweets, dev_labels)
dev_sampler = SequentialSampler(dev_data)
dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=batch_size)

test_data = TensorDataset(test_tweets, test_labels)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)



# Now, it's time to train and evaluate.

# Accuracy function
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=-1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)



# Function for evaluating test data.
def eval(model, dataloader, eval_type="Validation"):
    # Evaluate once every training epoch.
    
    # Put model into eval mode
    model.eval()
    
    # Loss, accuracy, steps, confusion matrix
    eval_loss, eval_accuracy = 0, 0
    num_eval_steps = 0
    cm = np.zeros((2, 2))

    # For each batch, run through model
    for batch in tqdm(dataloader, desc=eval_type):

        # Put batch on CUDA
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_labels = batch

        print(b_input_ids)
        print(b_labels)

        with torch.no_grad():
            b_encoding = model(b_input_ids, labels=b_labels)
            tmp_eval_loss, logits = b_encoding[:2]
            tmp_eval_loss = tmp_eval_loss.mean() # for some reason it spits out 2 losses lol

        # Evaluate on CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        tmp_eval_accuracy = flat_accuracy(logits, label_ids)

        # Update loss, accuracy, steps
        eval_loss += tmp_eval_loss.item()
        eval_accuracy += tmp_eval_accuracy
        num_eval_steps += 1

        # Get the actual predictions (0, 1, 2)
        predictions = np.argmax(logits, axis=-1)

        print(predictions)

        # Create the confusion matrix
        cm += confusion_matrix(label_ids, predictions, labels=range(2))

    # Calculate recall, precision, F1 :)
    recall = np.diag(cm) * 1.0 / (np.sum(cm, axis=1) + 10 ** -10)
    precision = np.diag(cm) * 1.0 / (np.sum(cm, axis=0) + 10 ** -10)
    f1 = 2 * precision * recall / (precision + recall + 10 ** -10)
    eval_loss = eval_loss / num_eval_steps

    print("{} loss: {}".format(eval_type, eval_loss), flush=True)
    print("{} accuracy: {}".format(eval_type, eval_accuracy / num_eval_steps), flush=True)

    for label in range(len(categories)):
        print("{} Precision: {}".format(categories[label], precision[label]), flush=True)
        print("{} Recall: {}".format(categories[label], recall[label]), flush=True)
        print("{} F1: {}".format(categories[label], f1[label]), flush=True)

    return eval_accuracy / num_eval_steps



# Training function.
# Epoch 0 is initial, i.e. no training
def train(model):
    print("- - - Training - - -")

    # Set CUDA
    if cuda == True:
        # Parallelization on multiple CUDA
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        # Turn on CUDA
        model.to(device)

    # Set optimizer
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)

    # Begin training
    for epoch_index in trange(epochs, desc="Epoch"):

        # Only fine-tune on epoch > 0
        torch.set_grad_enabled(epoch_index > 0)

        if epoch_index > 0:
            model.train()

        # Keep track of loss, steps
        total_loss, num_steps = 0., 0
        best_dev_acc = 0.

        for step, batch in enumerate(tqdm(train_dataloader, desc="Train")):

            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_labels = batch

            # Forward pass
            b_encoding = model(b_input_ids, labels=b_labels)
            loss, logits = b_encoding[:2]
            loss = loss.mean() # for some reason it spits out 2 losses lol
            print("Loss: {}".format(loss))
            print("Logits: {}".format(logits))

            # Update loss, steps
            total_loss += loss.item()
            num_steps += 1

            if epoch_index > 0:
                # Backward pass
                loss.backward()
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
                # Update parameters
                optimizer.step()
                optimizer.zero_grad()

        # Print train loss per epoch
        print("Train loss: {}".format(total_loss / num_steps), flush=True)

        # Evaluate once per epoch
        dev_acc = eval(model, dev_dataloader, eval_type="Validation")
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            test_acc = eval(model, test_dataloader, eval_type="Test")

        # Save model weights each epoch.
        if isinstance(model, nn.DataParallel):
            model.module.save_pretrained('./model_weights')
        else:
            model.save_pretrained('./model_weights')


# Boom
train(model)