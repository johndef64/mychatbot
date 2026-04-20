"""
image_gen.py — Image generation helpers for MyChatbot.
Supports: OpenAI (DALL-E), Google (Gemini/Imagen), OpenRouter, Alibaba (DashScope).

Alibaba model families (updated April 2026 from official docs):
  - WanX legacy   : wanx2.0-t2i-turbo, wanx2.1-t2i-turbo/plus
                    → old ImageSynthesis endpoint
  - Wan 2.2-2.5   : wan2.2-t2i-flash, wan2.2-t2i-plus, wan2.5-t2i-preview
                    → old ImageSynthesis endpoint
  - Wan 2.6+      : wan2.6-t2i, wan2.7-image, wan2.7-image-pro
                    → NEW multimodal-generation endpoint
  - Qwen-Image    : qwen-image, qwen-image-plus, qwen-image-max, etc.
                    → NEW multimodal-generation endpoint
  - Z-Image       : z-image-turbo
                    → NEW multimodal-generation endpoint
"""
import os
import re
import base64
import random
from io import BytesIO
from datetime import datetime
from PIL import Image, PngImagePlugin

# ── Model catalogues ──────────────────────────────────────────────────────────

OPENAI_IMAGE_MODELS = {
    "dall-e-3 (best)":       "dall-e-3",
    "dall-e-2 (cheapest)":   "dall-e-2",
    "gpt-image-1":           "gpt-image-1",
}

GOOGLE_IMAGE_MODELS = {
    "gemini-2.5-flash-image (cheapest)": "gemini-2.5-flash-image",
    "gemini-3.1-flash-image-preview":    "gemini-3.1-flash-image-preview",
    "gemini-3-pro-image-preview":        "gemini-3-pro-image-preview",
    "imagen-4.0-fast (fast)":            "imagen-4.0-fast-generate-001",
    "imagen-4.0 (quality)":              "imagen-4.0-generate-001",
    "imagen-4.0-ultra":                  "imagen-4.0-ultra-generate-001",
}
GOOGLE_IMAGEN_IDS = {
    "imagen-4.0-fast-generate-001",
    "imagen-4.0-generate-001",
    "imagen-4.0-ultra-generate-001",
}

OPENROUTER_IMAGE_MODELS = {
    "flux.2-klein-4b (cheapest)":        "black-forest-labs/flux.2-klein-4b",
    "gemini-3.1-flash-image (fast)":     "google/gemini-3.1-flash-image-preview",
    "gemini-2.5-flash-image":            "google/gemini-2.5-flash-image",
    "flux.2-flex":                       "black-forest-labs/flux.2-flex",
    "flux.2-pro":                        "black-forest-labs/flux.2-pro",
    "flux.2-max":                        "black-forest-labs/flux.2-max",
    "gpt-5-image-mini":                  "openai/gpt-5-image-mini",
    "seedream-4.5":                      "bytedance-seed/seedream-4.5",
    "riverflow-v2-fast":                 "sourceful/riverflow-v2-fast",
}

# ── Alibaba DashScope image models (updated April 2026) ───────────────────────
#
# Two distinct API protocols on DashScope:
#
#   LEGACY  → POST /services/aigc/text2image/image-synthesis  (async poll)
#             Models: wanx2.0-t2i-turbo, wanx2.1-t2i-turbo, wanx2.1-t2i-plus,
#                     wan2.2-t2i-flash, wan2.2-t2i-plus, wan2.5-t2i-preview
#
#   NEW     → POST /services/aigc/multimodal-generation/generation  (sync)
#             Models: wan2.6-t2i, wan2.7-image, wan2.7-image-pro,
#                     qwen-image, qwen-image-plus, qwen-image-max,
#                     qwen-image-max-2025-12-30, qwen-image-plus-2026-01-09,
#                     z-image-turbo

ALIBABA_IMAGE_MODELS = {
    # ── WanX legacy (cheapest, old protocol) ──────────────────────────────────
    # "wanx2.0-t2i-turbo (cheapest)":         "wanx2.0-t2i-turbo",
    # "wanx2.1-t2i-turbo":                    "wanx2.1-t2i-turbo",
    # "wanx2.1-t2i-plus":                     "wanx2.1-t2i-plus",
    # ── Wan 2.2 / 2.5 (old protocol) ─────────────────────────────────────────
    "wan2.2-t2i-flash (fast)":              "wan2.2-t2i-flash",
    "wan2.2-t2i-plus":                      "wan2.2-t2i-plus",
    "wan2.5-t2i-preview":                   "wan2.5-t2i-preview",
    # ── Wan 2.6 / 2.7 (new protocol) ─────────────────────────────────────────
    "wan2.6-t2i":                           "wan2.6-t2i",
    "wan2.7-image (fast)":                  "wan2.7-image",
    "wan2.7-image-pro (4K, best)":          "wan2.7-image-pro",
    # ── Qwen-Image (new protocol, best text rendering) ────────────────────────
    "qwen-image-max (recommended)":         "qwen-image-max",
    "qwen-image-plus":                      "qwen-image-plus",
    "qwen-image":                           "qwen-image",
    "qwen-image-plus-2026-01-09 (fast)":    "qwen-image-plus-2026-01-09",
    # ── Z-Image (new protocol, lightweight) ───────────────────────────────────
    "z-image-turbo (lightweight)":          "z-image-turbo",
}

# Models that use the OLD legacy ImageSynthesis endpoint
ALIBABA_LEGACY_MODEL_IDS = {
    "wanx2.0-t2i-turbo",
    "wanx2.1-t2i-turbo",
    "wanx2.1-t2i-plus",
    "wan2.2-t2i-flash",
    "wan2.2-t2i-plus",
    "wan2.5-t2i-preview",
}

# Default (cheapest) model per provider
DEFAULT_IMAGE_MODELS = {
    "OpenAI":     "dall-e-2 (cheapest)",
    "Google":     "gemini-2.5-flash-image (cheapest)",
    "OpenRouter": "flux.2-klein-4b (cheapest)",
    "Alibaba":    "wanx2.0-t2i-turbo (cheapest)",
}

# All image trigger keywords
IMAGE_TRIGGERS = {"#create", "#imagine", "#draw", "#paint", "#generate", "#image"}


def is_image_prompt(text: str) -> tuple[bool, str]:
    """Return (True, cleaned_prompt) if text starts with an image trigger word."""
    low = text.strip().lower()
    for trigger in IMAGE_TRIGGERS:
        if low.startswith(trigger):
            cleaned = text.strip()[len(trigger):].strip()
            return True, cleaned
    return False, text


def save_image(image: Image.Image, prompt: str, model_id: str,
               folder: str = "images") -> str:
    """Save PIL image to folder, return the file path."""
    os.makedirs(folder, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_prompt = re.sub(r'[^\w\-]', '_', prompt[:40])
    model_short = model_id.split("/")[-1]
    filename = f"{safe_prompt}__{model_short}__{ts}.png"
    path = os.path.join(folder, filename)

    # Embed metadata
    meta = PngImagePlugin.PngInfo()
    meta.add_text("Prompt", prompt)
    meta.add_text("Model", model_id)
    meta.add_text("Timestamp", ts)
    image.save(path, "PNG", pnginfo=meta)
    return path


# ── OpenAI DALL-E ─────────────────────────────────────────────────────────────

def generate_openai(prompt: str, api_key: str, model_label: str,
                    size: str = "1024x1024") -> tuple[Image.Image | None, str | None]:
    """Generate image with OpenAI DALL-E. Returns (PIL Image, error_str)."""
    from openai import OpenAI
    model_id = OPENAI_IMAGE_MODELS.get(model_label, model_label)
    try:
        client = OpenAI(api_key=api_key)
        if model_id == "dall-e-3":
            resp = client.images.generate(model=model_id, prompt=prompt,
                                          size=size, quality="standard", n=1)
        elif model_id == "gpt-image-1":
            resp = client.images.generate(model=model_id, prompt=prompt,
                                          size=size, n=1)
        else:  # dall-e-2
            resp = client.images.generate(model=model_id, prompt=prompt,
                                          size="1024x1024", n=1)
        # Response can be URL or b64_json
        item = resp.data[0]
        if hasattr(item, "b64_json") and item.b64_json:
            img = Image.open(BytesIO(base64.b64decode(item.b64_json)))
        elif hasattr(item, "url") and item.url:
            import requests
            r = requests.get(item.url, timeout=30)
            r.raise_for_status()
            img = Image.open(BytesIO(r.content))
        else:
            return None, "No image data in response"
        return img, None
    except Exception as e:
        return None, str(e)


# ── Google Gemini / Imagen ────────────────────────────────────────────────────

def generate_google(prompt: str, api_key: str, model_label: str,
                    aspect_ratio: str = "1:1") -> tuple[Image.Image | None, str | None]:
    """Generate image with Google AI Studio (Gemini or Imagen)."""
    model_id = GOOGLE_IMAGE_MODELS.get(model_label, model_label)
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        seed = random.randint(1, 999999)

        if model_id in GOOGLE_IMAGEN_IDS:
            # Imagen API
            resp = client.models.generate_images(
                model=model_id,
                prompt=prompt,
                config=types.GenerateImagesConfig(
                    number_of_images=1,
                    aspect_ratio=aspect_ratio,
                    seed=seed,
                    safety_filter_level="BLOCK_NONE",
                )
            )
            if resp.generated_images:
                img = Image.open(BytesIO(resp.generated_images[0].image.image_bytes))
                return img, None
            return None, "No image returned by Imagen"

        # Gemini image generation
        safety = [
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH",       threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT",        threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
        ]
        resp = client.models.generate_content(
            model=model_id,
            contents=[prompt],
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
                safety_settings=safety,
                seed=seed,
                image_config=types.ImageConfig(aspect_ratio=aspect_ratio),
            )
        )
        if resp.candidates and resp.candidates[0].content.parts:
            for part in resp.candidates[0].content.parts:
                if part.inline_data is not None:
                    img = Image.open(BytesIO(part.inline_data.data))
                    return img, None
        return None, "No image part in Gemini response"

    except ImportError:
        return None, "google-genai package not installed. Run: pip install google-genai"
    except Exception as e:
        return None, str(e)


# ── OpenRouter ────────────────────────────────────────────────────────────────

def generate_openrouter(prompt: str, api_key: str, model_label: str,
                        aspect_ratio: str = "1:1") -> tuple[Image.Image | None, str | None]:
    """Generate image via OpenRouter image-capable models."""
    from openai import OpenAI
    model_id = OPENROUTER_IMAGE_MODELS.get(model_label, model_label)
    try:
        client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
        seed = random.randint(1, 999999)

        is_gemini = "gemini" in model_id.lower()
        extra = {"image_config": {"aspect_ratio": aspect_ratio}}
        if is_gemini:
            extra = {
                "modalities": ["image", "text"],
                "image_config": {"aspect_ratio": aspect_ratio, "image_size": "1K"},
                "safety_settings": [
                    {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH",       "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                ]
            }

        resp_full = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
            seed=seed,
            extra_body=extra,
        )
        msg = resp_full.choices[0].message

        # Method 1: response.images attribute
        if hasattr(msg, "images") and msg.images:
            url = msg.images[0]["image_url"]["url"]
            enc = url.split(",", 1)[1] if "," in url else url
            return Image.open(BytesIO(base64.b64decode(enc))), None

        # Method 2: content list with image_url parts
        if hasattr(msg, "content") and isinstance(msg.content, list):
            for part in msg.content:
                ptype = part.get("type") if isinstance(part, dict) else getattr(part, "type", None)
                if ptype == "image_url":
                    url = (part.get("image_url", {}).get("url", "")
                           if isinstance(part, dict)
                           else part.image_url.url)
                    enc = url.split(",", 1)[1] if "," in url else url
                    return Image.open(BytesIO(base64.b64decode(enc))), None

        # Method 3: base64 embedded in string content
        if hasattr(msg, "content") and isinstance(msg.content, str):
            m = re.search(r'data:image/[^;]+;base64,([A-Za-z0-9+/=]+)', msg.content)
            if m:
                return Image.open(BytesIO(base64.b64decode(m.group(1)))), None

        return None, f"No image found in response (finish={resp_full.choices[0].finish_reason})"

    except Exception as e:
        return None, str(e)


# ── Alibaba DashScope ─────────────────────────────────────────────────────────

def _alibaba_size(aspect_ratio: str, model_id: str) -> str:
    """Map aspect ratio to the correct size string for the given model family."""
    if model_id in ALIBABA_LEGACY_MODEL_IDS:
        # Old ImageSynthesis endpoint uses width*height format
        size_map = {
            "1:1":  "1024*1024",
            "16:9": "1280*720",
            "9:16": "720*1280",
            "4:3":  "1280*960",
            "3:4":  "960*1280",
        }
        return size_map.get(aspect_ratio, "1024*1024")
    elif model_id in ("wan2.7-image-pro", "wan2.7-image"):
        # wan2.7 uses 1K / 2K / 4K spec or explicit pixels
        ar_map = {
            "1:1":  "2K",
            "16:9": "1920*1088",
            "9:16": "1088*1920",
            "4:3":  "1792*1344",
            "3:4":  "1344*1792",
        }
        return ar_map.get(aspect_ratio, "2K")
    else:
        # wan2.6, qwen-image*, z-image-turbo → explicit pixel sizes
        ar_map = {
            "1:1":  "1280*1280",
            "16:9": "1280*720",
            "9:16": "720*1280",
            "4:3":  "1280*960",
            "3:4":  "960*1280",
        }
        return ar_map.get(aspect_ratio, "1280*1280")


def _alibaba_legacy(prompt: str, api_key: str, model_id: str,
                    size: str) -> tuple[Image.Image | None, str | None]:
    """
    Call the OLD DashScope text2image endpoint (async poll).
    Used for: wanx2.0-t2i-turbo, wanx2.1-*, wan2.2-*, wan2.5-t2i-preview
    """
    import requests as req, time

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
        "X-DashScope-Async": "enable",
    }
    body = {
        "model": model_id,
        "input": {"prompt": prompt},
        "parameters": {"size": size, "n": 1, "prompt_extend": True, "watermark": False},
    }
    try:
        r = req.post(
            "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/text2image/image-synthesis",
            json=body, headers=headers, timeout=15,
        )
        r.raise_for_status()
        task_id = r.json()["output"]["task_id"]
    except Exception as e:
        return None, f"Task creation failed: {e}"

    # Poll for result (max ~90 s)
    for _ in range(30):
        time.sleep(3)
        try:
            poll = req.get(
                f"https://dashscope-intl.aliyuncs.com/api/v1/tasks/{task_id}",
                headers={"Authorization": f"Bearer {api_key}"}, timeout=10,
            )
            poll.raise_for_status()
            data = poll.json()
            status = data["output"]["task_status"]
            if status == "SUCCEEDED":
                url = data["output"]["results"][0]["url"]
                img_r = req.get(url, timeout=30)
                img_r.raise_for_status()
                return Image.open(BytesIO(img_r.content)), None
            elif status in ("FAILED", "CANCELED"):
                return None, f"Task {status}: {data['output'].get('message', '')}"
        except Exception as e:
            return None, f"Polling error: {e}"
    return None, "Timeout waiting for Alibaba image generation"


def _alibaba_new(prompt: str, api_key: str, model_id: str,
                 size: str) -> tuple[Image.Image | None, str | None]:
    """
    Call the NEW DashScope multimodal-generation endpoint (synchronous).
    Used for: wan2.6-t2i, wan2.7-image*, qwen-image*, z-image-turbo
    """
    import requests as req

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }
    body = {
        "model": model_id,
        "input": {
            "messages": [
                {
                    "role": "user",
                    "content": [{"text": prompt}],
                }
            ]
        },
        "parameters": {
            "size": size,
            "n": 1,
            "prompt_extend": True,
            "watermark": False,
        },
    }
    try:
        r = req.post(
            "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation",
            json=body, headers=headers, timeout=120,
        )
        r.raise_for_status()
        data = r.json()

        # Extract image URL from response
        choices = data.get("output", {}).get("choices", [])
        if choices:
            content = choices[0].get("message", {}).get("content", [])
            for item in content:
                if item.get("type") == "image" or "image" in item:
                    url = item.get("image") or item.get("url")
                    if url:
                        img_r = req.get(url, timeout=30)
                        img_r.raise_for_status()
                        return Image.open(BytesIO(img_r.content)), None

        return None, f"No image in response: {data}"
    except Exception as e:
        return None, str(e)


def generate_alibaba(prompt: str, api_key: str, model_label: str,
                     aspect_ratio: str = "1:1") -> tuple[Image.Image | None, str | None]:
    """
    Generate image with Alibaba DashScope.
    Automatically routes to the correct API protocol based on the model.
    """
    model_id = ALIBABA_IMAGE_MODELS.get(model_label, model_label)
    size = _alibaba_size(aspect_ratio, model_id)

    if model_id in ALIBABA_LEGACY_MODEL_IDS:
        return _alibaba_legacy(prompt, api_key, model_id, size)
    else:
        return _alibaba_new(prompt, api_key, model_id, size)


# ── Unified entry point ───────────────────────────────────────────────────────

def generate_image(prompt: str, provider: str, model_label: str,
                   api_keys: dict, aspect_ratio: str = "1:1") -> tuple[Image.Image | None, str | None]:
    """
    Dispatch image generation to the right provider.
    Returns (PIL Image or None, error string or None).
    """
    if provider == "OpenAI":
        key = api_keys.get("openai", "")
        size_map = {"1:1": "1024x1024", "16:9": "1792x1024", "9:16": "1024x1792"}
        size = size_map.get(aspect_ratio, "1024x1024")
        return generate_openai(prompt, key, model_label, size)

    elif provider == "Google":
        key = api_keys.get("googleai", api_keys.get("gemini", ""))
        return generate_google(prompt, key, model_label, aspect_ratio)

    elif provider == "OpenRouter":
        key = api_keys.get("openrouter", "")
        return generate_openrouter(prompt, key, model_label, aspect_ratio)

    elif provider == "Alibaba":
        key = api_keys.get("alibaba", "")
        return generate_alibaba(prompt, key, model_label, aspect_ratio)

    return None, f"Provider '{provider}' does not support image generation"