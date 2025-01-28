import torch
from data_processing.utils import prepare_example, get_minibatch, prepare_minibatch


def simple_evaluate(model, data, prep_fn=prepare_example, **kwargs):
    """Accuracy of a model on given data set."""
    correct = 0
    total = 0
    model.eval() # disable dropout

    for example in data:
        # convert the example input and label to PyTorch tensors
        x, target = prep_fn(example, model.vocab)

        # forward pass without backpropagation (no_grad)
        # get the output from the neural network for input x
        with torch.no_grad():
            logits = model(x)

        prediction = logits.argmax(dim=-1) # get the prediction

        # add the number of correct predictions to the total correct
        correct += (prediction == target).sum().item()
        total += 1

    return correct, total, correct / float(total)


def evaluate(
    model, data, batch_fn=get_minibatch, prep_fn=prepare_minibatch, batch_size=16
):
    """Accuracy of a model on given data set (using mini-batches)"""
    correct = 0
    total = 0
    model.eval() # disable dropout

    for mb in batch_fn(data, batch_size=batch_size, shuffle=False):
        x, targets = prep_fn(mb, model.vocab)
        with torch.no_grad():
            logits = model(x)

        predictions = logits.argmax(dim=-1).view(-1)

        # add the number of correct predictions to the total correct
        correct += (predictions == targets.view(-1)).sum().item()
        total += targets.size(0)

    return correct, total, correct / float(total)
