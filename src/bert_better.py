import torch, gc
import torch.nn as nn
import pandas as pd
import numpy as np
from transformers import *
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from utils import flat_accuracy
from evaluation import eval
from training import train

import click



@click.command()
@click.option('--actions-data', required=True, type=str, help='CSV filename containing conservation actions')
@click.option('--batch-size', type=int, default=32, help='Batch size for training')
@click.option('--cuda-num', type=int, default=None, help='Number of GPU to run on')
@click.option('--test-size', type=float, default=0.2, help='% test in train/test split')
@click.option('--epochs', type=int, default=10, help='Number of epochs to train')
@click.option('--learning-rate', type=float, default=2e-5, help='Learning rate for training')
@click.option('--adam-epsilon', type=float, default=1e-8, help='Adam epsilon')
@click.option('--weight-decay', type=float, default=0.0, help='Weight decay')
@click.option('--max-grad-norm', type=float, default=1.0, help='Maximum gradient norm')
def main(actions_data: str, batch_size: int, cuda_num: int, test_size: float,
            epochs: int, learning_rate: float, adam_epsilon: float, weight_decay: float, max_grad_norm: float):
    # 1. Set up GPU
    gc.collect()
    torch.cuda.empty_cache()
    if cuda_num:
        device = torch.device("cuda:{}".format(cuda_num) if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device: {}'.format(device))
    n_gpu = torch.cuda.device_count()

    # 2. Load in model
    categories = ['negative', 'neutral', 'positive']
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(categories))

    # 3. Load in dataset
    df = pd.read_csv(actions_data)

    print(df)

    # 4. Extract sentences and labels
    actions = []
    labels = []

    for i in range(len(df)):
        action = df.iloc[i]['action'][:512]
        label = df.iloc[i]['effectiveness_number']
        action_tokens = tokenizer.encode(action)
        padding = [0] * (512 - len(action_tokens))
        action_tokens += padding
        actions.append(action_tokens)
        labels.append(label)
    
    # Set up train/test data

    train_X, test_X, train_y, test_y = train_test_split(actions, labels, test_size=test_size)

    train_X = torch.tensor(train_X)
    test_X = torch.tensor(test_X)
    train_y = torch.tensor(train_y)
    test_y = torch.tensor(test_y)

    train_data = TensorDataset(train_X, train_y)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    
    test_data = TensorDataset(test_X, test_y)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

    # Train
    train(model=model,
            categories=categories,
            weight_decay=weight_decay,
            epochs=epochs,
            learning_rate=learning_rate,
            adam_epsilon=adam_epsilon,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            device=device)


if __name__ == '__main__':
    main()