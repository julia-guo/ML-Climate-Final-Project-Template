import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm, trange

from evaluation import eval

def train(model, categories, weight_decay, epochs, learning_rate, adam_epsilon, train_dataloader, test_dataloader, device):
    print("- - - Training - - -")
    print('model: {}'.format(model))
    print('Device: {}'.format(device))

    model.to(device)
    print('Device: {}'.format(device))

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
        print('Epoch {}'.format(epoch_index))

        # Only fine-tune on epoch > 0
        torch.set_grad_enabled(epoch_index > 0)

        if epoch_index > 0:
            model.train()

        # Keep track of loss, steps
        total_loss, num_steps = 0., 0

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
        eval(model=model, 
                dataloader=test_dataloader,
                categories=categories,
                device=device,
                eval_type="Test")

        # Save model weights each epoch.
        model.save_pretrained('./model_weights')