"""Generate text for evaluation with the project's trained models.

    model, tokenizer = load_model_and_tokenizer("runs/vanilla-test/checkpoints/best.pt")
    answer = generate(model, tokenizer, "Question: What is 1 + 1?\nAnswer:")

Load once and reuse the model/tokenizer across evaluation examples.
"""

import os

from .train import PROJECT_ROOT, configure_environment, project_path


configure_environment()
CACHE_DIR = PROJECT_ROOT / ".cache" / "huggingface"
os.environ["HF_HOME"] = str(CACHE_DIR)

import torch

from . import model as model_module


def load_model_and_tokenizer(checkpoint_path, device=None):
    """Load a training checkpoint and its GPT-2 tokenizer; prefer CUDA if available."""
    from transformers import GPT2TokenizerFast

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # Training checkpoints include Python/NumPy RNG state; use trusted project files.
    checkpoint = torch.load(project_path(checkpoint_path), map_location="cpu", weights_only=False)
    config = checkpoint["model_config"]
    model_args = {key: value for key, value in config.items()
                  if key not in {"name", "use_model", "use_weighted_from_each_loop", "use_regularizer"}}
    model = getattr(model_module, config["use_model"])(**model_args)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    tokenizer = GPT2TokenizerFast.from_pretrained(
        "openai-community/gpt2", cache_dir=str(CACHE_DIR)
    )
    tokenizer.model_max_length = model.max_seq_len
    return model, tokenizer


@torch.inference_mode()
def generate(model, tokenizer, prompt, max_new_tokens=128, temperature=0.0,
             max_seq_len=None):
    """Return only the generated continuation of a single text prompt.

    temperature=0 uses greedy decoding; positive values enable sampling.
    Stop on EOS or max_new_tokens. Each step uses the last max_seq_len tokens
    as context (default: the model's limit). This cannot exceed the model's
    position embedding limit. An empty prompt starts from the EOS token.
    """
    if max_new_tokens < 0 or temperature < 0:
        raise ValueError("max_new_tokens and temperature must be non-negative")
    max_seq_len = model.max_seq_len if max_seq_len is None else max_seq_len
    if not 1 <= max_seq_len <= model.max_seq_len:
        raise ValueError(f"max_seq_len must be between 1 and {model.max_seq_len}")

    # Context truncation happens below, so tokenizer length warnings do not apply.
    token_ids = tokenizer.encode(prompt, add_special_tokens=False, verbose=False)
    if not token_ids:
        token_ids = [tokenizer.eos_token_id]
    tokens = torch.tensor([token_ids], dtype=torch.long, device=next(model.parameters()).device)
    generated = []
    was_training = model.training
    model.eval()
    try:
        for _ in range(max_new_tokens):
            logits = model(tokens[:, -max_seq_len:])[:, -1, :]
            if temperature == 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                probabilities = torch.softmax(logits.float() / temperature, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)
            token_id = next_token.item()
            if token_id == tokenizer.eos_token_id:
                break
            generated.append(token_id)
            tokens = torch.cat((tokens, next_token), dim=1)
    finally:
        model.train(was_training)
    return tokenizer.decode(generated, skip_special_tokens=True)
