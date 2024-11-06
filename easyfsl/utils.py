"""
General utilities
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision
from matplotlib import pyplot as plt
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from easyfsl.methods import FewShotClassifier


def plot_images(images: Tensor, title: str, images_per_row: int):
    """
    Plot images in a grid.
    Args:
        images: 4D mini-batch Tensor of shape (B x C x H x W)
        title: title of the figure to plot
        images_per_row: number of images in each row of the grid
    """
    plt.figure()
    plt.title(title)
    plt.imshow(
        torchvision.utils.make_grid(images, nrow=images_per_row).permute(1, 2, 0)
    )


def sliding_average(value_list: List[float], window: int) -> float:
    """
    Computes the average of the latest instances in a list
    Args:
        value_list: input list of floats (can't be empty)
        window: number of instances to take into account. If value is 0 or greater than
            the length of value_list, all instances will be taken into account.

    Returns:
        average of the last window instances in value_list

    Raises:
        ValueError: if the input list is empty
    """
    if len(value_list) == 0:
        raise ValueError("Cannot perform sliding average on an empty list.")
    return np.asarray(value_list[-window:]).mean()


def predict_embeddings(
    dataloader: DataLoader,
    model: nn.Module,
    device: Optional[str] = None,
) -> pd.DataFrame:
    """
    Predict embeddings for a dataloader.
    Args:
        dataloader: dataloader to predict embeddings for. Must deliver tuples (images, class_names)
        model: model to use for prediction
        device: device to cast the images to. If none, no casting is performed. Must be the same as
            the device the model is on.
    Returns:
        dataframe with columns embedding and class_name
    """
    all_embeddings = []
    all_class_names = []
    with torch.no_grad():
        for images, class_names in tqdm(
            dataloader, unit="batch", desc="Predicting embeddings"
        ):
            if device is not None:
                images = images.to(device)
            all_embeddings.append(model(images).detach().cpu())
            if isinstance(class_names, torch.Tensor):
                all_class_names += class_names.tolist()
            else:
                all_class_names += class_names

    concatenated_embeddings = torch.cat(all_embeddings)

    return pd.DataFrame(
        {"embedding": list(concatenated_embeddings), "class_name": all_class_names}
    )

from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
import numpy as np

from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score
import numpy as np

# Existing utility to evaluate on one task, modified to return predictions and true labels
def evaluate_on_one_task(
    model: FewShotClassifier,
    support_images: Tensor,
    support_labels: Tensor,
    query_images: Tensor,
    query_labels: Tensor,
) -> Tuple[int, int, Tensor, Tensor]:
    """
    Returns the number of correct predictions of query labels, the total number of
    predictions, and the predictions and true labels for further metrics calculation.
    """
    model.process_support_set(support_images, support_labels)
    predictions = model(query_images).detach().data
    predicted_labels = torch.max(predictions, 1)[1]
    number_of_correct_predictions = int((predicted_labels == query_labels).sum().item())
    return number_of_correct_predictions, len(query_labels), predicted_labels, query_labels

# Modified evaluate function to include multiple metrics
def evaluate(
    model: FewShotClassifier,
    data_loader: DataLoader,
    device: str = "cuda",
    use_tqdm: bool = True,
    tqdm_prefix: Optional[str] = None,
) -> dict:
    """
    Evaluate the model on few-shot classification tasks with multiple metrics.
    Args:
        model: a few-shot classifier
        data_loader: loads data in the shape of few-shot classification tasks
        device: where to cast data tensors.
            Must be the same as the device hosting the model's parameters.
        use_tqdm: whether to display the evaluation's progress bar
        tqdm_prefix: prefix of the tqdm bar
    Returns:
        A dictionary containing average classification accuracy, sensitivity, specificity, F1 score, and AUC-ROC.
    """
    # Initialize counters and lists to accumulate metrics
    total_predictions = 0
    correct_predictions = 0

    all_true_labels = []
    all_predicted_labels = []
    all_predicted_probs = []  # To store prediction probabilities for AUC-ROC

    # Set model to evaluation mode
    model.eval()
    with torch.no_grad():
        # Use a tqdm context to show a progress bar in the logs
        with tqdm(
            enumerate(data_loader),
            total=len(data_loader),
            disable=not use_tqdm,
            desc=tqdm_prefix,
        ) as tqdm_eval:
            for _, (
                support_images,
                support_labels,
                query_images,
                query_labels,
                _,
            ) in tqdm_eval:
                # Get predictions and true labels
                correct, total, predicted_labels, true_labels = evaluate_on_one_task(
                    model,
                    support_images.to(device),
                    support_labels.to(device),
                    query_images.to(device),
                    query_labels.to(device),
                )

                total_predictions += total
                correct_predictions += correct

                # Append labels for metric calculation
                all_true_labels.extend(true_labels.cpu().numpy())
                all_predicted_labels.extend(predicted_labels.cpu().numpy())
                all_predicted_probs.extend(model(query_images.to(device)).softmax(dim=1).cpu().numpy())

                # Log accuracy in real time
                tqdm_eval.set_postfix(accuracy=correct_predictions / total_predictions)

    # Convert all labels to numpy arrays for metric calculation
    all_true_labels = np.array(all_true_labels)
    all_predicted_labels = np.array(all_predicted_labels)
    all_predicted_probs = np.array(all_predicted_probs)

    # Compute confusion matrix to derive sensitivity and specificity
    if len(np.unique(all_true_labels)) == 2:
        # Binary classification case
        cm = confusion_matrix(all_true_labels, all_predicted_labels)
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        # Multi-class case: calculate sensitivity and specificity for each class
        sensitivity = []
        specificity = []
        cm = confusion_matrix(all_true_labels, all_predicted_labels)
        for i in range(len(cm)):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = cm.sum() - (tp + fn + fp)
            sensitivity.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
            specificity.append(tn / (tn + fp) if (tn + fp) > 0 else 0.0)
        sensitivity = np.mean(sensitivity)
        specificity = np.mean(specificity)

    # Compute other metrics
    accuracy = correct_predictions / total_predictions
    f1 = f1_score(all_true_labels, all_predicted_labels, average='weighted')
    precision = precision_score(all_true_labels, all_predicted_labels, average='weighted')
    recall = recall_score(all_true_labels, all_predicted_labels, average='weighted')

    # AUC-ROC score (for both binary and multi-class classification)
    try:
        if len(np.unique(all_true_labels)) == 2:
            auc_roc = roc_auc_score(all_true_labels, all_predicted_probs[:, 1])
        else:
            auc_roc = roc_auc_score(all_true_labels, all_predicted_probs, multi_class='ovr', average='weighted')
    except ValueError:
        auc_roc = None  # AUC-ROC cannot be computed if there is only one class in true labels

    metrics = {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "auc_roc": auc_roc,
    }

    return metrics

# Example usage
# metrics = evaluate(model, data_loader)
# print(metrics)

def compute_average_features_from_images(
    dataloader: DataLoader,
    model: nn.Module,
    device: Optional[str] = None,
):
    """
    Compute the average features vector from all images in a DataLoader.
    Assumes the images are always first element of the batch.
    Returns:
        Tensor: shape (1, feature_dimension)
    """
    all_embeddings = torch.stack(
        predict_embeddings(dataloader, model, device)["embedding"].to_list()
    )
    average_features = all_embeddings.mean(dim=0)
    if device is not None:
        average_features = average_features.to(device)
    return average_features
