import os
import gc
import torch
import numpy as np
import folder_paths
import hashlib

# ─── Global model cache ────────────────────────────────────────────────────────
_MODEL_CACHE: dict = {}

BACKBONE_MODELS = [
    "neuphonic/neutts-air",
    "neuphonic/neutts-air-q8-gguf",
    "neuphonic/neutts-air-q4-gguf",
    "neuphonic/neutts-nano",
    "neuphonic/neutts-nano-q8-gguf",
    "neuphonic/neutts-nano-q4-gguf",
    "neuphonic/neutts-nano-french",
    "neuphonic/neutts-nano-french-q8-gguf",
    "neuphonic/neutts-nano-french-q4-gguf",
    "neuphonic/neutts-nano-german",
    "neuphonic/neutts-nano-german-q8-gguf",
    "neuphonic/neutts-nano-german-q4-gguf",
    "neuphonic/neutts-nano-spanish",
    "neuphonic/neutts-nano-spanish-q8-gguf",
    "neuphonic/neutts-nano-spanish-q4-gguf",
]

CODEC_MODELS = [
    "neuphonic/neucodec",
    "neuphonic/neucodec-onnx-decoder",
]

SAMPLE_RATE = 24000


def _get_device_options():
    opts = ["cpu"]
    if torch.cuda.is_available():
        opts.insert(0, "cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        opts.append("mps")
    return opts


def _load_tts(backbone_repo: str, backbone_device: str, codec_repo: str, codec_device: str):
    cache_key = f"{backbone_repo}|{backbone_device}|{codec_repo}|{codec_device}"
    if cache_key in _MODEL_CACHE:
        print(f"✅ [NeuTTS] Using cached model: {backbone_repo}")
        return _MODEL_CACHE[cache_key]

    try:
        from neutts import NeuTTS
    except ImportError:
        raise ImportError(
            "neutts not installed. Run: pip install neutts\n"
            "For GGUF support: pip install neutts[llama]\n"
            "For ONNX codec: pip install neutts[onnx]"
        )

    print(f"⏳ [NeuTTS] Loading backbone={backbone_repo} on {backbone_device} | codec={codec_repo} on {codec_device}")

    # Allow local model paths under ComfyUI/models/neutts/
    neutts_model_dir = os.path.join(folder_paths.models_dir, "neutts")
    local_backbone = os.path.join(neutts_model_dir, backbone_repo.replace("/", os.sep))
    local_codec = os.path.join(neutts_model_dir, codec_repo.replace("/", os.sep))

    resolved_backbone = local_backbone if os.path.isdir(local_backbone) else backbone_repo
    resolved_codec = local_codec if os.path.isdir(local_codec) else codec_repo

    tts = NeuTTS(
        backbone_repo=resolved_backbone,
        backbone_device=backbone_device,
        codec_repo=resolved_codec,
        codec_device=codec_device,
    )

    _MODEL_CACHE[cache_key] = tts
    print(f"✅ [NeuTTS] Model loaded and cached.")
    return tts


def _unload_models():
    keys = list(_MODEL_CACHE.keys())
    print(f"🗑️  [NeuTTS] Unloading {len(keys)} cached model(s)...")
    for k in keys:
        del _MODEL_CACHE[k]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("✅ [NeuTTS] Model cache cleared.")


def _audio_to_comfy(wav: np.ndarray) -> tuple:
    """Convert numpy wav array → ComfyUI AUDIO tuple (waveform_tensor, sample_rate)."""
    if wav.ndim == 1:
        wav = wav[np.newaxis, :]          # (1, samples)
    waveform = torch.from_numpy(wav).float()
    if waveform.ndim == 2:
        waveform = waveform.unsqueeze(0)  # (1, channels, samples)
    return ({"waveform": waveform, "sample_rate": SAMPLE_RATE},)


# ─── Node 1: NeuTTS (Basic TTS + Voice Clone) ─────────────────────────────────

class NeuTTSNode:
    """
    Synthesise speech from text using a reference audio for voice cloning.
    Works like Qwen-TTS VoiceCloneNode but uses NeuTTS on-device models.
    """

    CATEGORY = "audio/NeuTTS"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"

    @classmethod
    def INPUT_TYPES(cls):
        devices = _get_device_options()
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "Hello, this is NeuTTS running inside ComfyUI!"}),
                "backbone_model": (BACKBONE_MODELS, {"default": BACKBONE_MODELS[0]}),
                "codec_model": (CODEC_MODELS, {"default": CODEC_MODELS[0]}),
                "backbone_device": (devices, {"default": devices[0]}),
                "codec_device": (devices, {"default": devices[0]}),
                "unload_after_generate": ("BOOLEAN", {"default": False}),
            },
            # ref_text kept in required so it renders as a widget (not a socket slot),
            # keeping slot indices predictable: slot 0 = ref_audio, slot 1 = ref_codes
            "optional": {
                "ref_audio": ("AUDIO",),
                "ref_codes": ("NEUTTS_REF_CODES",),
                "ref_text": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    def generate(
        self,
        text: str,
        backbone_model: str,
        codec_model: str,
        backbone_device: str,
        codec_device: str,
        unload_after_generate: bool,
        ref_audio=None,
        ref_text: str = "",
        ref_codes=None,
    ):
        tts = _load_tts(backbone_model, backbone_device, codec_model, codec_device)

        resolved_ref_codes = ref_codes  # pre-encoded takes priority

        if resolved_ref_codes is None and ref_audio is not None:
            import tempfile, soundfile as sf
            waveform = ref_audio["waveform"].squeeze(0).numpy()
            sr = ref_audio["sample_rate"]
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, waveform.T if waveform.ndim == 2 else waveform, sr)
                tmp_path = tmp.name
            print(f"⏳ [NeuTTS] Encoding reference audio...")
            resolved_ref_codes = tts.encode_reference(tmp_path)
            os.unlink(tmp_path)

        print(f"⏳ [NeuTTS] Generating speech...")
        wav = tts.infer(text, resolved_ref_codes, ref_text if ref_text.strip() else None)

        if unload_after_generate:
            _unload_models()

        return _audio_to_comfy(wav)


# ─── Node 2: NeuTTS Encode Reference ─────────────────────────────────────────

class NeuTTSEncodeRefNode:
    """
    Pre-encode a reference audio once and reuse across multiple NeuTTSNode
    calls. Saves time when cloning the same voice repeatedly.
    Like Qwen-TTS VoiceClonePromptNode.
    """

    CATEGORY = "audio/NeuTTS"
    RETURN_TYPES = ("NEUTTS_REF_CODES",)
    RETURN_NAMES = ("ref_codes",)
    FUNCTION = "encode"

    @classmethod
    def INPUT_TYPES(cls):
        devices = _get_device_options()
        return {
            "required": {
                "ref_audio": ("AUDIO",),
                "backbone_model": (BACKBONE_MODELS, {"default": BACKBONE_MODELS[0]}),
                "codec_model": (CODEC_MODELS, {"default": CODEC_MODELS[0]}),
                "backbone_device": (devices, {"default": devices[0]}),
                "codec_device": (devices, {"default": devices[0]}),
            },
        }

    def encode(self, ref_audio, backbone_model, backbone_device, codec_model, codec_device):
        import tempfile, soundfile as sf

        tts = _load_tts(backbone_model, backbone_device, codec_model, codec_device)

        waveform = ref_audio["waveform"].squeeze(0).numpy()
        sr = ref_audio["sample_rate"]
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, waveform.T if waveform.ndim == 2 else waveform, sr)
            tmp_path = tmp.name

        print(f"⏳ [NeuTTS] Encoding reference audio...")
        ref_codes = tts.encode_reference(tmp_path)
        os.unlink(tmp_path)
        print(f"✅ [NeuTTS] Reference encoded.")
        return (ref_codes,)


# ─── Node 3: NeuTTS Model Unloader ────────────────────────────────────────────

class NeuTTSUnloadModels:
    """Force-clear all cached NeuTTS models to free VRAM/RAM."""

    CATEGORY = "audio/NeuTTS"
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "unload"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    def unload(self):
        _unload_models()
        return {}


# ─── Node mapping ─────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "NeuTTSNode": NeuTTSNode,
    "NeuTTSEncodeRefNode": NeuTTSEncodeRefNode,
    "NeuTTSUnloadModels": NeuTTSUnloadModels,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NeuTTSNode": "NeuTTS 🎙️",
    "NeuTTSEncodeRefNode": "NeuTTS Encode Reference 🎤",
    "NeuTTSUnloadModels": "NeuTTS Unload Models 🗑️",
}
