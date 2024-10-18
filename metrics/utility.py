from transformers import AutoConfig, PretrainedConfig

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
