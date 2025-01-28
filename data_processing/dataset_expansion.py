import argparse
import logging
from nltk import Tree

from .utils import get_datasets


# Logging setup
logging.basicConfig(
    level=logging.INFO, format="%(process)d - %(levelname)s - %(message)s"
)


def tokens_from_treestring(treestring):
    "Get the leaves from a tree string after parsing the Tree."

    tr = Tree.fromstring(treestring)
    return tr.leaves()

def extract_subtrees(tree):
    "Recursively extract all subtrees from the given tree, including the tree itself."
    
    # Yield the current tree
    yield tree
    # For each child that is also a Tree, recurse
    for child in tree:
        if isinstance(child, Tree):
            yield from extract_subtrees(child)

def write_all_subtrees(data, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for example in data:
            # Extract subtrees
            for subtree in extract_subtrees(example.tree):
                # Convert the subtree back to the bracketed string format
                # Using str(subtree) gives a bracketed representation similar to the input.
                subtree_str = " ".join(str(subtree).split())  # normalize spacing
                f.write(subtree_str + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract all subtrees from dataset training file.")
    parser.add_argument(
        "--base_path",
        type=str,
        default="trees/",
        help="Base path for input and output file. Default is 'trees/'."
    )

    args = parser.parse_args()
    base_path = args.base_path

    train_data, dev_data, test_data = get_datasets(base_path)

    write_all_subtrees(train_data, base_path + "train_all_subtrees.txt")
    logging.info(f"Wrote train_all_subtrees.txt to {base_path}!")
