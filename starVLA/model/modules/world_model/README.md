# World-Model Backends

`starVLA/model/modules/world_model/` contains physical-world pretraining
backends that can condition action prediction. Their wrapper contract mirrors
the VLM interface so frameworks can select a backend without changing the
dataloader or deployment boundary.

## Interface

| Method | Contract |
| --- | --- |
| `__init__(config)` | Load the pretrained model and processor. |
| `forward(**kwargs)` | Run the model forward pass. |
| `generate(**kwargs)` | Run autoregressive generation when supported. |
| `build_qwenvl_inputs(images, instructions)` | Build model inputs. |

```python
from starVLA.model.modules.world_model import get_world_model

world_model = get_world_model(config)
```

Current implementations include Cosmos-Reason2 through a Qwen3-VL-compatible
wrapper and Cosmos-Predict2 through a diffusion-transformer wrapper. The
matching Qwen3-VL architecture is why Cosmos-Reason2 can share the standard VLM
processor and hidden-state contract while retaining its physical-reasoning
pretraining.

These generic world-model backends are separate from the repository's two
active action/world research projects:

- Dual-Query WAM is implemented by `QwenGR00T` and its WAM guidance modules.
- Action-World Co-Flow is implemented by `QwenActionWorldCoFlow` and
  `action_world_coflow/`.

Keep framework-specific coupling in those framework packages rather than in
this backend layer.
