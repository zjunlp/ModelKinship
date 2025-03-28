from transformers import AutoConfig, PretrainedConfig, AutoModelForCausalLM, AutoTokenizer
import torch
import logging
from tqdm import tqdm

def get_config(model: str, trust_remote_code: bool = False) -> PretrainedConfig:
    """
    Fetch the configuration of a pretrained model from HuggingFace.

    Args:
        model (str): The name or path of the model to load configuration for.
        trust_remote_code (bool, optional): Whether to trust remote code during loading.
                                            Defaults to False.

    Returns:
        PretrainedConfig: The configuration object of the specified model.
    """
    # Fetch the configuration from HuggingFace's model hub.
    config = AutoConfig.from_pretrained(
        model,
        trust_remote_code=trust_remote_code,  # Whether to allow remote code execution.
    )
    return config


def validate_models(
        model_1: str,
        model_2: str,
        base_model: str
) -> None:
    """
    Validate model names to ensure they are different and exist.

    Args:
        model_1: Name of the first model
        model_2: Name of the second model
        base_model: Name of the base model

    Raises:
        click.BadParameter: If validation fails
    """
    if model_1 == model_2 or model_1 == base_model or model_2 == base_model:
        raise click.BadParameter("All model names must be different")


def quantize_8bit(x: torch.Tensor) -> torch.Tensor:
    # Get min and max values
    x_min, x_max = x.min(), x.max()

    # Create 256 evenly spaced levels between min and max (8-bit = 2^8 = 256 levels)
    levels = torch.linspace(x_min, x_max, 256, device=x.device)

    # For each value in x, find the closest level
    # Using torch.bucketize for efficient binning
    indices = torch.bucketize(x, levels).clamp(0, 255)

    # Return the quantized values
    return levels[indices]


def extract_delta_parameters(
        model_1_name: str,
        model_2_name: str,
        model_base_name: str,
        low_precision: bool,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> (torch.Tensor, torch.Tensor):

    """
    Extract the delta parameters (weight differences) between two models
    relative to a base model.

    Args:
        model_1_name (str): Name or path of the first model.
        model_2_name (str): Name or path of the second model.
        model_base_name (str): Name or path of the base model for comparison.
        low_precision (bool): Whether to use low precision weights

    Returns:
        (torch.Tensor, torch.Tensor): Delta parameters of model_1 and model_2 relative to base model.
    """

    # Extract state dictionaries from models
    model_1 = AutoModelForCausalLM.from_pretrained(model_1_name).to(device)
    state_dict_1 = model_1.state_dict()
    del model_1

    model_2 = AutoModelForCausalLM.from_pretrained(model_2_name).to(device)
    state_dict_2 = model_2.state_dict()
    del model_2

    model_base = AutoModelForCausalLM.from_pretrained(model_base_name).to(device)
    state_dict_base = model_base.state_dict()
    del model_base

    # Determine the number of layers
    num_layers = state_dict_base['lm_head.weight'].shape[0]

    # Check if model architectures match, log a warning if not
    if state_dict_1['lm_head.weight'].shape[0] != state_dict_2['lm_head.weight'].shape[0]:
        shape_1 = state_dict_1['lm_head.weight'].shape
        shape_2 = state_dict_2['lm_head.weight'].shape
        logging.warning(f'Warning: Model architectures do not match. '
                        f'Using sub weight space instead.\n'
                        f'lm_head.weight shape in model 1: {shape_1}, '
                        f'lm_head.weight shape in model 2: {shape_2}')

    # Initialize lists to store delta parameters for both models
    d_vector_1, d_vector_2 = [], []

    # Iterate over keys in the base model's state dictionary with tqdm
    for key, base_params in tqdm(state_dict_base.items(), desc="Processing keys", unit="key"):
        # Only proceed if key exists in both models
        try:
            if key not in state_dict_1 or key not in state_dict_2:
                logging.warning(f'Key {key} not found in one of the models')
                continue
        except Exception as e:
            logging.error(f'Error processing key {key}: {str(e)}')

        # Get the parameters for each model (truncate to num_layers for consistency)
        params_1 = state_dict_1[key][:num_layers]
        params_2 = state_dict_2[key][:num_layers]

        # Compute the deltas relative to the base model
        delta_1 = (params_1 - base_params).view(-1)
        delta_2 = (params_2 - base_params).view(-1)

        # Accumulate deltas
        d_vector_1.append(delta_1)
        d_vector_2.append(delta_2)

    # Clear memory
    del state_dict_1, state_dict_2, state_dict_base

    d_vector_1 = torch.cat(d_vector_1)
    d_vector_2 = torch.cat(d_vector_2)

    if low_precision:
        d_vector_1 = quantize_8bit(d_vector_1)
        d_vector_2 = quantize_8bit(d_vector_2)

    else:
        return d_vector_1, d_vector_2
