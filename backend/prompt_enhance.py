import threading
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import settings


_BASE_SYSTEM_PROMPT = (
    "You are a prompt engineer for image and video diffusion models. "
    "Rewrite the user's short description as a detailed, vivid prompt suitable for FLUX or Wan-style models. "
    "Include: subject and action, composition and framing, lighting, color palette, materials and textures, "
    "mood, and a stylistic descriptor (photographic, illustrated, cinematic, etc.) when ambiguous. "
    "Do not invent specific named people. Do not add NSFW content. "
    "Output only the rewritten prompt - no preamble, no explanation, no quotes. Keep it under 80 words."
)

_VIDEO_SYSTEM_PROMPT = (
    "You are a prompt engineer for video diffusion models that animate or replace characters in existing footage. "
    "Rewrite the user's short description as a detailed prompt focused on the character's motion, "
    "scene, lighting, camera framing, and mood. Avoid static composition language. "
    "Do not invent specific named people. Do not add NSFW content. "
    "Output only the rewritten prompt - no preamble, no explanation, no quotes. Keep it under 80 words."
)


def _system_prompt_for(target_model: str) -> str:
    if target_model in {"character-swap", "scail-2char"}:
        return _VIDEO_SYSTEM_PROMPT
    return _BASE_SYSTEM_PROMPT


_model = None
_tokenizer = None
_load_lock = threading.Lock()


def _load() -> None:
    global _model, _tokenizer
    if _model is not None:
        return
    with _load_lock:
        if _model is not None:
            return
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not available. Verify torch was installed with the CUDA wheel "
                "(pip install torch --index-url https://download.pytorch.org/whl/cu124)."
            )
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        _tokenizer = AutoTokenizer.from_pretrained(settings.QWEN_MODEL_ID)
        _model = AutoModelForCausalLM.from_pretrained(
            settings.QWEN_MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            dtype=torch.bfloat16,
        )
        _model.eval()


def enhance(prompt: str, target_model: str = "text-to-image") -> str:
    _load()
    assert _tokenizer is not None and _model is not None
    messages = [
        {"role": "system", "content": _system_prompt_for(target_model)},
        {"role": "user", "content": prompt},
    ]
    text = _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _tokenizer(text, return_tensors="pt").to(_model.device)
    with torch.inference_mode():
        out = _model.generate(
            **inputs,
            max_new_tokens=400,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=_tokenizer.eos_token_id,
        )
    generated = out[0][inputs["input_ids"].shape[1]:]
    result = _tokenizer.decode(generated, skip_special_tokens=True).strip()
    if result.startswith('"') and result.endswith('"'):
        result = result[1:-1].strip()
    return result
