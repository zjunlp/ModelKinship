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
):
    state_dict_1 = AutoModelForCausalLM.from_pretrained(model_1_name).state_dict()
    state_dict_2 = AutoModelForCausalLM.from_pretrained(model_2_name).state_dict()
    state_dict_base = AutoModelForCausalLM.from_pretrained(model_base_name).state_dict()
    num_layers = state_dict_1['lm_head.weight'].shape[0]

    if state_dict_1['lm_head.weight'].shape[0] != state_dict_2['lm_head.weight'].shape[0]:
        logging.warning('Warning: Model architecture not match. Use sub weight space instead')

    # Calculate the delta parameter
    d_vector_1, d_vector_2 = [], []
    for key in AutoModelForCausalLM.from_pretrained(model_base_name).state_dict():
        if key in state_dict_1:
            d_vector_1.append((state_dict_1[key][:num_layers] - state_dict_base[key]).reshape(-1))
            d_vector_2.append((state_dict_2[key][:num_layers] - state_dict_base[key]).reshape(-1))

    # release memory
    del state_dict_1, state_dict_2, state_dict_base

    return torch.cat(d_vector_1), torch.cat(d_vector_2)


def calculate_metric(d_vector_1, d_vector_2, metric):
    # Pearson Correlation Coefficient
    if metric == 'pcc':
        stack = torch.stack((d_vector_1, d_vector_2), dim=0)
        return torch.corrcoef(stack)[0][1]
    # Euclidean Distance
    elif metric == 'ed':
        distance = torch.dist(d_vector_1, d_vector_2)
        return distance
    # Cosine Similarity
    elif metric == 'cs':
        cs = cosine_similarity(d_vector_1, d_vector_2)
        return cs
    else:
        return 0