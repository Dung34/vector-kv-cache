"""
Model and Tokenizer Loading Utilities.
Configures PyTorch precision, device placement, and attention implementation.
"""

from typing import Tuple, Any, Optional
import torch

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    AutoModelForCausalLM = None
    AutoTokenizer = None


def load_model_and_tokenizer(
    model_id: str = "Qwen/Qwen2.5-1.5B-Instruct",
    device: Optional[str] = None,
    torch_dtype: Optional[torch.dtype] = None,
    attn_implementation: str = "eager",
) -> Tuple[Any, Any]:
    """
    Loads pretrained CausalLM model and tokenizer with eager attention implementation
    for explicit attention weights access during H2O hook interception.
    """
    if AutoModelForCausalLM is None or AutoTokenizer is None:
        raise ImportError("Hugging Face transformers package is not installed. Please run: pip install transformers accelerate")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if torch_dtype is None:
        torch_dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading tokenizer for '{model_id}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model '{model_id}' on {device} (dtype={torch_dtype}, attn={attn_implementation})...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map=device if device != "cpu" else None,
        attn_implementation=attn_implementation,
        trust_remote_code=True,
    )
    if device == "cpu":
        model = model.to("cpu")

    model.eval()
    return model, tokenizer
