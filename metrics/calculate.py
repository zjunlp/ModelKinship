import logging
import torch
import numpy


def cosine_similarity(a, b):
    similarity = numpy.sqrt(numpy.dot(a, b) ** 2 / (numpy.dot(a, a) * numpy.dot(b, b)))
    return similarity


def calculate_metric(d_vector_1: torch.Tensor, d_vector_2: torch.Tensor, metric: str) -> str:
    """
    Calculate the specified metric between two delta vectors.

    Args:
        d_vector_1 (torch.Tensor): Delta parameters for model 1.
        d_vector_2 (torch.Tensor): Delta parameters for model 2.
        metric (str): The metric to calculate ('pcc', 'ed', 'cs').

    Returns:
        str: A formatted string with the result of the chosen metric.
    """

    # Pearson Correlation Coefficient (PCC)
    if metric == 'pcc':
        # Stack the two vectors and calculate the Pearson correlation coefficient
        stack = torch.stack((d_vector_1, d_vector_2), dim=0)
        pcc = torch.corrcoef(stack)[0, 1].item()
        return f"Model Kinship based on Pearson Correlation Coefficient: {pcc}"

    # Euclidean Distance (ED)
    elif metric == 'ed':
        # Compute the Euclidean distance between the vectors
        distance = torch.dist(d_vector_1, d_vector_2).item()
        return f"Model Kinship based on Euclidean Distance: {distance}"

    # Cosine Similarity (CS)
    elif metric == 'cs':
        # Compute cosine similarity
        cs = cosine_similarity(d_vector_1, d_vector_2)
        return f"Model Kinship based on Cosine Similarity: {cs}"

    # If metric is not recognized
    else:
        return "Invalid metric specified."

