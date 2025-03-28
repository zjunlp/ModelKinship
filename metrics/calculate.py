import logging
import torch
import numpy
from tqdm import tqdm

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
    # Pearson Correlation Coefficient (PCC)
    if metric == 'pcc':
        mean_1 = torch.mean(d_vector_1)
        mean_2 = torch.mean(d_vector_2)

        # Compute covariance with tqdm
        covariance = sum((x - mean_1) * (y - mean_2) for x, y in tqdm(zip(d_vector_1, d_vector_2),
                                                                      total=len(d_vector_1),
                                                                      desc="Computing Covariance"))

        std_1 = torch.sqrt(sum((x - mean_1) ** 2 for x in tqdm(d_vector_1, desc="Computing Std Dev Model 1")))
        std_2 = torch.sqrt(sum((y - mean_2) ** 2 for y in tqdm(d_vector_2, desc="Computing Std Dev Model 2")))

        pcc = (covariance / (std_1 * std_2)).item()
        return f"Model Kinship based on Pearson Correlation Coefficient: {pcc}"

    # Euclidean Distance (ED)
    elif metric == 'ed':
        distance = torch.sqrt(sum((x - y) ** 2 for x, y in tqdm(zip(d_vector_1, d_vector_2),
                                                                total=len(d_vector_1),
                                                                desc="Calculating Euclidean Distance"))).item()
        return f"Model Kinship based on Euclidean Distance: {distance}"

    # Cosine Similarity (CS)
    elif metric == 'cs':
        dot_product = sum(x * y for x, y in tqdm(zip(d_vector_1, d_vector_2),
                                                 total=len(d_vector_1),
                                                 desc="Computing Cosine Similarity"))
        norm_1 = torch.sqrt(sum(x ** 2 for x in d_vector_1))
        norm_2 = torch.sqrt(sum(y ** 2 for y in d_vector_2))
        cs = (dot_product / (norm_1 * norm_2)).item()
        return f"Model Kinship based on Cosine Similarity: {cs}"


    # If metric is not recognized
    else:
        return "Invalid metric specified."

