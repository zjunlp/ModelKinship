import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy


def cosine_similarity(a, b):
    similarity = numpy.sqrt(numpy.dot(a, b) ** 2 / (numpy.dot(a, a) * numpy.dot(b, b)))
    return similarity


def extract_delta_parameters(
        model_1_name: str,
        model_2_name: str,
        model_base_name: str,
) -> (torch.Tensor, torch.Tensor):
    """
    Extract the delta parameters (weight differences) between two models
    relative to a base model.

    Args:
        model_1_name (str): Name or path of the first model.
        model_2_name (str): Name or path of the second model.
        model_base_name (str): Name or path of the base model for comparison.

    Returns:
        (torch.Tensor, torch.Tensor): Delta parameters of model_1 and model_2 relative to base model.
    """
    # Load the state dictionaries for each model
    state_dict_1 = AutoModelForCausalLM.from_pretrained(model_1_name).state_dict()
    state_dict_2 = AutoModelForCausalLM.from_pretrained(model_2_name).state_dict()
    state_dict_base = AutoModelForCausalLM.from_pretrained(model_base_name).state_dict()

    # Determine the number of layers
    num_layers = state_dict_base['lm_head.weight'].shape[0]

    # Check if model architectures match, log a warning if not
    if state_dict_1['lm_head.weight'].shape[0] != state_dict_2['lm_head.weight'].shape[0]:
        logging.warning('Warning: Model architectures do not match. Using sub weight space instead.')

    # Initialize lists to store delta parameters for both models
    d_vector_1, d_vector_2 = [], []

    # Iterate over keys in the base model's state dictionary
    for key, base_params in state_dict_base.items():
        # Only proceed if key exists in both models
        if key in state_dict_1 and key in state_dict_2:
            # Get the parameters for each model (truncate to num_layers for consistency)
            params_1 = state_dict_1[key][:num_layers]
            params_2 = state_dict_2[key][:num_layers]

            # Compute the deltas relative to the base model
            delta_1 = (params_1 - base_params).view(-1)
            delta_2 = (params_2 - base_params).view(-1)

            # Accumulate deltas
            d_vector_1.append(delta_1)
            d_vector_2.append(delta_2)

    d_vector_1 = torch.cat(d_vector_1)
    d_vector_2 = torch.cat(d_vector_2)

    # Clear memory of unused variables
    del state_dict_1, state_dict_2, state_dict_base

    return d_vector_1, d_vector_2


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

