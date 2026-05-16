# ComfyUI-NeuTTS

ComfyUI custom nodes for **[NeuTTS](https://github.com/neuphonic/neutts)** — Neuphonic's on-device TTS with instant voice cloning.

---

## Nodes

| Node | Description |
|------|-------------|
| **NeuTTS 🎙️** | TTS + voice cloning. Takes text + optional ref audio → generates speech |
| **NeuTTS Encode Reference 🎤** | Pre-encode a reference audio once; reuse across multiple NeuTTS nodes |
| **NeuTTS Unload Models 🗑️** | Force-clear all cached models to free GPU/RAM |

---

## Installation

### Via ComfyUI Manager
Search `ComfyUI-NeuTTS` → Install.

### Manual
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/YOUR_NAME/ComfyUI-NeuTTS
cd ComfyUI-NeuTTS
pip install -r requirements.txt
```

### Install NeuTTS
```bash
# CPU only
pip install neutts

# GGUF support (llama-cpp-python)
pip install neutts[llama]

# ONNX codec (faster decode)
pip install neutts[onnx]

# Everything
pip install neutts[all]
```

For GPU-accelerated llama-cpp-python (CUDA):
```bash
CMAKE_ARGS="-DGGML_CUDA=ON" pip install "neutts[llama]" --force-reinstall --no-cache-dir
```

---

## Model Directory (optional local cache)

Pre-download models to avoid HuggingFace timeouts:

```
ComfyUI/
└── models/
    └── neutts/
        └── neuphonic/
            ├── neutts-air/
            ├── neutts-nano/
            └── neucodec/
```

The nodes auto-detect local models before falling back to HuggingFace Hub.

---

## Available Backbones

| Model | Size | Lang | Notes |
|-------|------|------|-------|
| `neuphonic/neutts-air` | ~552M | EN | Best quality |
| `neuphonic/neutts-air-q4-gguf` | Q4 | EN | Fastest, needs llama-cpp |
| `neuphonic/neutts-nano` | ~229M | EN | Fast |
| `neuphonic/neutts-nano-q4-gguf` | Q4 | EN | CPU-friendly |
| `neuphonic/neutts-nano-french` | ~229M | FR | |
| `neuphonic/neutts-nano-german` | ~229M | DE | |
| `neuphonic/neutts-nano-spanish` | ~229M | ES | |

Codec: `neuphonic/neucodec` (default) or `neuphonic/neucodec-onnx-decoder` (faster).

---

## Reference Audio Tips

- Format: `.wav`, mono, 16–44 kHz
- Length: **3–15 seconds**
- Content: clean speech, no background noise
- Providing `ref_text` (transcript of the ref clip) significantly improves quality

---

## Workflow Example

```
Load Audio → [NeuTTS Encode Reference] → ref_codes ─┐
                                                     ├→ [NeuTTS] → audio → Preview Audio
                     text ────────────────────────────┘
```

---

## Acknowledgements

- [NeuTTS](https://github.com/neuphonic/neutts) by [Neuphonic](https://neuphonic.com)
- Inspired by [ComfyUI-Qwen-TTS](https://github.com/flybirdxx/ComfyUI-Qwen-TTS)

## License

Apache 2.0
