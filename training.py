import torch
from torch import nn
import logging
import time
import os
from data_processing.utils import get_datasets, prepare_example, get_examples
from evaluate import simple_evaluate
from models import TreeLSTMClassifier
from experiment import Experiment


# Logging setup
logging.basicConfig(
    level=logging.INFO, format="%(process)d - %(levelname)s - %(message)s"
)


CHECKPOINT_BASE_PATH = "artifacts/checkpoints"


def train_model(
    model,
    optimizer,
    data: tuple,
    num_iterations=10000,
    print_every=1000,
    eval_every=1000,
    batch_fn=get_examples,
    prep_fn=prepare_example,
    eval_fn=simple_evaluate,
    batch_size=1,
    eval_batch_size=None,
    experiment: Experiment = None,
):
    """Train a model."""
    iter_i = 0
    train_loss = 0.0
    print_num = 0
    start = time.time()
    criterion = nn.CrossEntropyLoss()  # loss function
    best_eval = 0.0
    best_iter = 0

    # store train loss and validation accuracy during training
    # so we can plot them afterwards
    losses = []
    accuracies = []

    ckpt_name = f"{model.__class__.__name__}_{model.seed}"
    ckpt_name = ("CAE" if hasattr(model, "use_cae") and model.use_cae else "") + ckpt_name
    ckpt_name = ("SubTree" if "SubTree" in experiment.model_name else "") + ckpt_name

    train_data, dev_data, test_data = data

    if eval_batch_size is None:
        eval_batch_size = batch_size

    directory_path = f"{CHECKPOINT_BASE_PATH}/{experiment.name}"
    os.makedirs(directory_path, exist_ok=True)

    while True:  # when we run out of examples, shuffle and continue
        for batch in batch_fn(train_data, batch_size=batch_size):

            # forward pass
            model.train()
            x, targets = prep_fn(batch, model.vocab)
            logits = model(x)

            B = targets.size(0)  # later we will use B examples per update

            # compute cross-entropy loss (our criterion)
            # note that the cross entropy loss function computes the softmax for us
            loss = criterion(logits.view([B, -1]), targets.view(-1))

            # Calculate and apply contractive regularization term in case of TreeLSTM with cae projection
            if isinstance(model, TreeLSTMClassifier) and model.use_cae:
                jacobian_loss = 0
                for proj_name in ["i_proj", "f_l_proj", "f_r_proj", "g_proj", "o_proj"]:
                    proj_layer = getattr(model.treelstm.reduce, proj_name)[
                        0
                    ]  # Get the Linear layer (0th element of nn.Sequential)
                    w_sum = torch.sum(proj_layer.weight**2, dim=1)
                    w_sum = w_sum.unsqueeze(1)
                    projection_output = getattr(
                        model.treelstm.reduce, proj_name.replace("proj", "cache")
                    )
                    if proj_name == "g_proj":
                        # This projection uses a TanH activation
                        dh_sq_tanh = (1 - projection_output**2) ** 2
                        jacobian_loss += torch.sum(torch.mm(dh_sq_tanh, w_sum), 0)
                    else:
                        # The other layers use a Sigmoid activation
                        dh_sq_sigmoid = (
                            projection_output * (1 - projection_output)
                        ) ** 2
                        jacobian_loss += torch.sum(torch.mm(dh_sq_sigmoid, w_sum), 0)

                logging.debug(
                    f"We have a loss of {loss.item()} and a Jacobian loss of {model.gamma * jacobian_loss.item()}."
                )
                loss += model.gamma * jacobian_loss.item()
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()

            # update weights - take a small step in the opposite dir of the gradient
            optimizer.step()

            print_num += 1
            iter_i += 1

            # print info
            if iter_i % print_every == 0:
                logging.info(
                    "Iter %r: loss=%.4f, time=%.2fs"
                    % (iter_i, train_loss, time.time() - start)
                )
                losses.append(train_loss)
                print_num = 0
                train_loss = 0.0

            # evaluate
            if iter_i % eval_every == 0:
                _, _, accuracy = eval_fn(
                    model,
                    dev_data,
                    batch_size=eval_batch_size,
                    batch_fn=batch_fn,
                    prep_fn=prep_fn,
                )
                accuracies.append(accuracy)
                logging.info("iter %r: dev acc=%.4f" % (iter_i, accuracy))

                # save best model parameters
                if accuracy > best_eval:
                    logging.info("new highscore")
                    best_eval = accuracy
                    best_iter = iter_i
                    path = f"{directory_path}/{ckpt_name}.pt"
                    ckpt = {
                        "state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_eval": best_eval,
                        "best_iter": best_iter,
                    }
                    torch.save(ckpt, path)

            # done training
            if iter_i == num_iterations:
                logging.info(f"Done training {model.__class__.__name__} with seed {model.seed}")

                # evaluate on train, dev, and test with best model
                logging.info("Loading best model")
                path = f"{directory_path}/{ckpt_name}.pt"
                ckpt = torch.load(path, weights_only=False)
                model.load_state_dict(ckpt["state_dict"])

                _, _, train_acc = eval_fn(
                    model,
                    train_data,
                    batch_size=eval_batch_size,
                    batch_fn=batch_fn,
                    prep_fn=prep_fn,
                )
                _, _, dev_acc = eval_fn(
                    model,
                    dev_data,
                    batch_size=eval_batch_size,
                    batch_fn=batch_fn,
                    prep_fn=prep_fn,
                )
                _, _, test_acc = eval_fn(
                    model,
                    test_data,
                    batch_size=eval_batch_size,
                    batch_fn=batch_fn,
                    prep_fn=prep_fn,
                )

                logging.info(
                    "best model iter {:d}: "
                    "train acc={:.4f}, dev acc={:.4f}, test acc={:.4f}".format(
                        best_iter, train_acc, dev_acc, test_acc
                    )
                )

                experiment.losses = losses
                experiment.accuracies = accuracies
                experiment.best_train_acc = train_acc
                experiment.best_dev_acc = dev_acc
                experiment.best_test_acc = test_acc

                return losses, accuracies
