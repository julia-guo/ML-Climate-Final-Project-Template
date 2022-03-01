import numpy as np
from tqdm import tqdm
import torch
from sklearn.metrics import confusion_matrix

from utils import flat_accuracy

def eval(model, dataloader, categories, device, eval_type="Validation"):
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