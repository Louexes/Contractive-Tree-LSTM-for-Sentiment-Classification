# This module contains all functions for data pre-processing to make the dataset available for model training.
from nltk import Tree
import torch
import numpy as np
from collections import namedtuple, Counter, OrderedDict
import re
import random
import logging


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# this function reads in a textfile and fixes an issue with "\\"
def filereader(path):
    with open(path, mode="r", encoding="utf-8") as f:
        for line in f:
            yield line.strip().replace("\\", "")


def tokens_from_treestring(s):
    "Extract the tokens from a sentiment tree."

    return re.sub(r"\([0-9] |\)", "", s).split()


def transitions_from_treestring(s):
    s = re.sub("\([0-5] ([^)]+)\)", "0", s)
    s = re.sub("\)", " )", s)
    s = re.sub("\([0-4] ", "", s)
    s = re.sub("\([0-4] ", "", s)
    s = re.sub("\)", "1", s)

    return list(map(int, s.split()))


Example = namedtuple("Example", ["tokens", "tree", "label", "transitions"])


def examplereader(path, lower=False):
    "Returns all examples in a file one by one."

    for line in filereader(path):
        line = line.lower() if lower else line
        tokens = tokens_from_treestring(line)
        tree = Tree.fromstring(line)
        label = int(line[1])
        trans = transitions_from_treestring(line)
        yield Example(tokens=tokens, tree=tree, label=label, transitions=trans)


def get_datasets(
    base_path: str = "trees", use_expanded_data: bool = False, lower: bool = False
) -> list[list]:
    """
    Get train, dev and test data.
    use_expanded_data is to select the data that contains all subtrees.
    """

    LOWER = lower  # we will keep the original casing by default
    if use_expanded_data:
        appendix = "_all_subtrees"  # "" for using the original dataset without all the subtrees.
    else:
        appendix = ""
    train_data = list(examplereader(f"{base_path}/train{appendix}.txt", lower=LOWER))
    dev_data = list(examplereader(f"{base_path}/dev.txt", lower=LOWER))
    test_data = list(examplereader(f"{base_path}/test.txt", lower=LOWER))

    return train_data, dev_data, test_data


def prepare_example(
    example, vocab, device: torch.device = DEVICE
) -> tuple[torch.LongTensor]:
    "Map tokens to their IDs for a single example."

    # vocab returns 0 if the word is not there (i2w[0] = <unk>)
    x = [vocab.w2i.get(t, 0) for t in example.tokens]

    x = torch.LongTensor([x])
    x = x.to(device)

    y = torch.LongTensor([example.label])
    y = y.to(device)

    return x, y


def get_examples(data, shuffle=True, **kwargs):
    "Shuffle data set and return 1 example at a time (until nothing left)"

    if shuffle:
        logging.info("Shuffling training data")
        random.shuffle(data)  # shuffle training data each epoch
    for example in data:
        yield example


### Mini-Batched versions below
def get_minibatch(data, batch_size=25, shuffle=True):
    "Return minibatches, optional shuffling."

    if shuffle:
        print("Shuffling training data")
        random.shuffle(data)  # shuffle training data each epoch

    batch = []
    for example in data:
        batch.append(example)

        if len(batch) == batch_size:
            yield batch
            batch = []

    # in case there is something left
    if len(batch) > 0:
        yield batch


def pad(tokens, length, pad_value=1):
    "Add padding 1s to a sequence to that it has the desired length."

    return tokens + [pad_value] * (length - len(tokens))


def prepare_minibatch(mb, vocab):
    """
    Minibatch is a list of examples.
    This function converts words to IDs and returns
    torch tensors to be used as input/targets.
    """

    maxlen = max([len(ex.tokens) for ex in mb])

    # vocab returns 0 if the word is not there
    x = [pad([vocab.w2i.get(t, 0) for t in ex.tokens], maxlen) for ex in mb]

    x = torch.LongTensor(x)
    x = x.to(DEVICE)

    y = [ex.label for ex in mb]
    y = torch.LongTensor(y)
    y = y.to(DEVICE)

    return x, y


### End of mini-batch functions


# Helper functions for batching and unbatching states
# For speed we want to combine computations by batching, but
# for processing logic we want to turn the output into lists again
# to easily manipulate.


def batch(states):
    """
    Turns a list of states into a single tensor for fast processing.
    This function also chunks (splits) each state into a (h, c) pair"""
    return torch.cat(states, 0).chunk(2, 1)


def unbatch(state):
    """
    Turns a tensor back into a list of states.
    First, (h, c) are merged into a single state.
    Then the result is split into a list of sentences.
    """
    return torch.split(torch.cat(state, 1), 1, 0)


def prepare_treelstm_minibatch(mb, vocab):
    """
    Returns sentences reversed (last word first)
    Returns transitions together with the sentences.
    """
    
    maxlen = max([len(ex.tokens) for ex in mb])

    # vocab returns 0 if the word is not there
    # NOTE: reversed sequence!
    x = [pad([vocab.w2i.get(t, 0) for t in ex.tokens], maxlen)[::-1] for ex in mb]

    x = torch.LongTensor(x)
    x = x.to(DEVICE)

    y = [ex.label for ex in mb]
    y = torch.LongTensor(y)
    y = y.to(DEVICE)

    maxlen_t = max([len(ex.transitions) for ex in mb])
    transitions = [pad(ex.transitions, maxlen_t, pad_value=2) for ex in mb]
    transitions = np.array(transitions)
    transitions = transitions.T  # time-major

    return (x, transitions), y


### Class definitions used in BOW
class OrderedCounter(Counter, OrderedDict):
    """Counter that remembers the order elements are first seen"""

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)


class Vocabulary:
    """A vocabulary, assigns IDs to tokens"""

    def __init__(self):
        self.freqs = OrderedCounter()
        self.w2i = {}
        self.i2w = []

    def count_token(self, t):
        self.freqs[t] += 1

    def add_token(self, t):
        self.w2i[t] = len(self.w2i)
        self.i2w.append(t)

    def build(self, min_freq=0):
        """
        min_freq: minimum number of occurrences for a word to be included
                  in the vocabulary
        """
        self.add_token("<unk>")  # reserve 0 for <unk> (unknown words)
        self.add_token("<pad>")  # reserve 1 for <pad> (discussed later)

        tok_freq = list(self.freqs.items())
        tok_freq.sort(key=lambda x: x[1], reverse=True)
        for tok, freq in tok_freq:
            if freq >= min_freq:
                self.add_token(tok)



### Helpers for sentence_length_comparison
def get_sentence_lengths(data):
    """
    Calculate the length of each sentence in the provided dataset.

    Args:
        data (list of dict): A list of dictionaries where each dictionary represents a sentence.
                             Each dictionary should have a key 'tokens' which maps to a list of tokens in the sentence.

    Returns:
        list of int: A list containing the length of each sentence (number of tokens).
    """

    lengths = []
    for example in data:
        tokens = example['tokens']
        lengths.append(len(tokens))
    return lengths

def bin_sentences(data, bins):
    """
    Bin sentences based on their lengths.

    Parameters:
    data (list of tuples): A list where each element is a tuple, and the first element of the tuple is a sentence (string).
    bins (list of int): A list of integers representing the bin edges. Sentences will be binned based on these edges.

    Returns:
    dict: A dictionary where the keys are bin indices and the values are lists of tuples (sentences) that fall into the corresponding bin.
    """

    binned_data = {i: [] for i in range(len(bins)-1)}
    for example in data:
        length = len(example[0])
        for i in range(len(bins)-1):
            if bins[i] < length <= bins[i+1]:
                binned_data[i].append(example)
                break
    return binned_data

def prepare_example_bow(example, vocab):
    """
    Prepares a bag-of-words (BoW) representation of an example using the given vocabulary.

    Args:
        example (object): An object containing the tokens and label of the example.
            - example.tokens (list of str): The tokens of the example.
            - example.label (int): The label of the example.
        vocab (object): An object containing the vocabulary with word-to-index mapping.
            - vocab.w2i (dict): A dictionary mapping words to their corresponding indices.

    Returns:
        tuple: A tuple containing:
            - x (torch.LongTensor): A tensor of shape (1, sequence_length) containing the indices of the tokens.
            - target (torch.Tensor): A tensor containing the label of the example.
    """

    x = torch.LongTensor([
        vocab.w2i.get(t, vocab.w2i["<unk>"]) for t in example.tokens 
    ]).unsqueeze(0)
    target = torch.tensor([example.label]) 
    return x, target

def prepare_example_lstm(example, vocab):
    """
    Prepares an example for LSTM input by converting tokens to their corresponding indices 
    and creating a target tensor for the label.

    Args:
        example (object): An object containing tokens and a label. 
                          `example.tokens` is a list of tokens.
                          `example.label` is the label for the example.
        vocab (object): A vocabulary object with a word-to-index mapping `w2i`.

    Returns:
        tuple: A tuple containing:
            - x (torch.LongTensor): A tensor of shape (1, sequence_length) with token indices.
            - target (torch.Tensor): A tensor containing the label.
    """
    
    x = torch.LongTensor([
        vocab.w2i.get(t, vocab.w2i["<unk>"]) for t in example.tokens 
    ]).unsqueeze(0)
    target = torch.tensor([example.label])
    return x, target


def prepare_example_tree_lstm(example, vocab):
    """
    Prepares input and target tensors for a Tree-LSTM model from a given example.

    Args:
        example (object): An example object containing tokens, label, and transitions.
        vocab (object): A vocabulary object with a word-to-index mapping.

    Returns:
        tuple: A tuple containing:
            - A tuple of input tensors:
                - x (torch.LongTensor): Tensor of token indices with shape (1, maxlen).
                - transitions (numpy.ndarray): Array of transitions with shape (maxlen_t, 1).
            - y (torch.LongTensor): Tensor of the example label with shape (1,).
    """
    
    # vocab returns 0 if the word is not there
    # NOTE: reversed sequence!
    x = [vocab.w2i.get(t, 0) for t in example.tokens][::-1]

    x = torch.LongTensor(x).unsqueeze(0)  # Add batch dimension
    x = x.to(DEVICE)

    y = torch.LongTensor([example.label])
    y = y.to(DEVICE)

    transitions = example.transitions
    transitions = np.array(transitions)
    transitions = transitions.reshape(-1, 1)  # time-major with batch dimension

    return (x, transitions), y

def evaluate_model_on_bin(model, data_bin, prep_fn):
    """
    Evaluate the performance of a model on a given data bin.
    Args:
        model (torch.nn.Module): The model to be evaluated.
        data_bin (iterable): A collection of data examples to evaluate the model on.
        prep_fn (function): A function that processes each example and returns the input tensor and target tensor.
    Returns:
        float or None: The accuracy of the model on the given data bin. Returns None if the data bin is empty.
    """

    correct = 0
    total = 0
    model.eval()
    
    with torch.no_grad():
        for example in data_bin:
            x, target = prep_fn(example, model.vocab)
            logits = model(x)
            # Handle models that return additional outputs (e.g., Tree LSTM)
            if isinstance(logits, tuple):
                logits = logits[0]
            prediction = logits.argmax(dim=-1)
            correct += int((prediction == target).item())
            total += 1
    
    if total > 0:
        accuracy = correct / total
    else:
        accuracy = None  # Handle empty bins if necessary
    return accuracy

def evaluate_model_on_bins(model, binned_data, prep_fn, bins):
    """
    Evaluates a given model on multiple bins of data and returns the performance metrics for each bin.

    Args:
        model: The model to be evaluated.
        binned_data (list): A list of data bins, where each bin contains a subset of the data.
        prep_fn (function): A function to preprocess the data before evaluation.

    Returns:
        list: A list of performance metrics (e.g., accuracy) for each data bin.
    """

    performances = []
    for i in range(len(bins)-1):
        data_bin = binned_data[i]
        accuracy = evaluate_model_on_bin(model, data_bin, prep_fn)
        performances.append(accuracy)
    return performances
