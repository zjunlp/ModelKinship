from transformers import AutoConfig, PretrainedConfig


def get_config(model, trust_remote_code: bool = False) -> PretrainedConfig:
    # Require connection to HuggingFace
    res = AutoConfig.from_pretrained(
        model,
        trust_remote_code=trust_remote_code,
    )
    return res