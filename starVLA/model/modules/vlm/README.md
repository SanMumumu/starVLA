# Vision-Language Model Backends

`starVLA/model/modules/vlm/` provides a uniform interface for the VLM backends
used by frameworks under `starVLA/model/framework/VLM4A/`.

## Interface

Each wrapper is an `nn.Module` that implements:

| Method | Contract |
| --- | --- |
| `__init__(config)` | Load the pretrained model and processor. |
| `forward(**kwargs)` | Run the training or inference forward pass. |
| `generate(**kwargs)` | Run autoregressive generation. |
| `build_qwenvl_inputs(images, instructions, solutions=None)` | Build model inputs. |

Use the factory instead of importing a concrete wrapper:

```python
from starVLA.model.modules.vlm import get_vlm_model

vlm = get_vlm_model(config)
```

The factory routes from `config.framework.qwenvl.base_vlm`. Current wrappers
cover Qwen2.5-VL, Qwen3-VL, Qwen3.5-VL, Florence-2, Gemma-4, Molmo2, and
MiniCPM-V. Cosmos world-model backends live in
`starVLA/model/modules/world_model/`; the factory delegates compatible Cosmos
names there to preserve legacy configs.

## Data flow

```text
framework.forward(raw examples)
  -> VLM wrapper builds processor inputs
  -> VLM emits hidden states
  -> action/world project heads consume selected hidden states
```

Training and deployment should both enter through the framework API so image,
state, action-normalization, and checkpoint contracts remain identical.
