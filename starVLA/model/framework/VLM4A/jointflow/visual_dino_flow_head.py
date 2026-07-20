"""Visual DINO flow-matching head built on the shared cross-attention DiT."""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta

from starVLA.model.modules.action_model.flow_matching_head.cross_attention_dit import DiT


_PREDICTION_TYPES = {"velocity", "jit_x"}
_FLOW_TIME_SAMPLING_TYPES = {"legacy", "gr00t"}


def visual_prediction_to_velocity(
    prediction: torch.Tensor,
    noisy_latent: torch.Tensor,
    t: torch.Tensor | float,
    *,
    prediction_type: str,
    t_eps: float,
) -> torch.Tensor:
    """Convert a visual-head output to the rectified-flow velocity.

    ``jit_x`` mirrors the action head's clean-sample parameterization: the
    network predicts the structured future DINO token directly, while Euler
    sampling still integrates a velocity field.
    """

    if prediction_type == "velocity":
        return prediction
    if prediction_type != "jit_x":
        raise ValueError(
            f"Unsupported visual prediction_type={prediction_type!r}; "
            f"expected one of {sorted(_PREDICTION_TYPES)}."
        )
    t = torch.as_tensor(t, device=noisy_latent.device, dtype=noisy_latent.dtype)
    return (prediction - noisy_latent) / (1.0 - t).clamp_min(float(t_eps))


######### // code // ##########
class VisualFlowMatchingHead(nn.Module):
    def __init__(self, full_config):
        super().__init__()
        cfg = full_config.framework.visual_model
        self.d_dino = int(cfg.get("d_dino", 384))
        self.hidden_size = int(cfg.get("hidden_size", 768))
        self.num_timestep_buckets = int(cfg.get("num_timestep_buckets", 1000))
        self.noise_s = float(cfg.get("noise_s", 0.999))
        self.num_inference_timesteps = int(cfg.get("num_inference_timesteps", 4))
        if self.num_inference_timesteps <= 0:
            raise ValueError(
                "visual_model.num_inference_timesteps must be positive, "
                f"got {self.num_inference_timesteps}"
            )
        self.prediction_type = str(cfg.get("prediction_type", "velocity")).lower()
        self.flow_time_sampling = str(cfg.get("flow_time_sampling", "legacy")).lower()
        self.jit_t_eps = float(cfg.get("jit_t_eps", 5.0e-2))
        self.clean_target_loss_weight = float(cfg.get("clean_target_loss_weight", 0.0))
        self.cosine_loss_weight = float(cfg.get("cosine_loss_weight", 0.0))
        if self.prediction_type not in _PREDICTION_TYPES:
            raise ValueError(
                f"visual_model.prediction_type must be one of {sorted(_PREDICTION_TYPES)}, "
                f"got {self.prediction_type!r}"
            )
        if self.flow_time_sampling not in _FLOW_TIME_SAMPLING_TYPES:
            raise ValueError(
                f"visual_model.flow_time_sampling must be one of {sorted(_FLOW_TIME_SAMPLING_TYPES)}, "
                f"got {self.flow_time_sampling!r}"
            )
        if self.jit_t_eps <= 0.0:
            raise ValueError(f"visual_model.jit_t_eps must be positive, got {self.jit_t_eps}")
        if self.clean_target_loss_weight < 0.0 or self.cosine_loss_weight < 0.0:
            raise ValueError(
                "visual clean/cosine loss weights must be non-negative, got "
                f"clean={self.clean_target_loss_weight}, cosine={self.cosine_loss_weight}"
            )
        self.add_pos_embed = bool(cfg.get("add_pos_embed", True))
        # WAM predicts one composite-image patch grid, not three separately encoded camera grids.
        # Keeping this bound explicit avoids allocating optimizer/gradient state for unused positional rows.
        self.max_target_tokens = int(cfg.get("max_target_tokens", cfg.get("max_seq_len", 1024)))
        if self.max_target_tokens <= 0:
            raise ValueError(f"max_target_tokens must be positive, got {self.max_target_tokens}")

        self.x_embed = nn.Linear(self.d_dino, self.hidden_size)
        self.x_decode = nn.Linear(self.hidden_size, self.d_dino)
        if self.add_pos_embed:
            self.position_embedding = nn.Embedding(self.max_target_tokens, self.hidden_size)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        dit_cfg = dict(cfg.get("diffusion_model_cfg", {}))
        dit_cfg.setdefault("num_attention_heads", int(cfg.get("num_attention_heads", 12)))
        dit_cfg.setdefault(
            "attention_head_dim",
            int(cfg.get("attention_head_dim", self.hidden_size // dit_cfg["num_attention_heads"])),
        )
        dit_cfg.setdefault("num_layers", int(cfg.get("num_layers", 8)))
        dit_cfg.setdefault("output_dim", self.hidden_size)
        dit_cfg.setdefault("dropout", float(cfg.get("dropout", 0.1)))
        dit_cfg.setdefault("final_dropout", True)
        dit_cfg.setdefault("interleave_self_attention", True)
        dit_cfg.setdefault("norm_type", "ada_norm")
        dit_cfg.setdefault("positional_embeddings", None)
        dit_cfg["cross_attention_dim"] = int(
            cfg.get("cross_attention_dim", full_config.framework.qwenvl.get("vl_hidden_dim", 896))
        )
        self.model = DiT(**dit_cfg)

        self.beta_dist = Beta(float(cfg.get("noise_beta_alpha", 1.5)), float(cfg.get("noise_beta_beta", 1.0)))

    #######
    @staticmethod
    def _module_dtype(module: nn.Module, fallback: torch.dtype = torch.float32) -> torch.dtype:
        for param in module.parameters(recurse=True):
            return param.dtype
        return fallback

    #######

    def sample_time(self, batch_size: int, device, dtype) -> torch.Tensor:
        if self.flow_time_sampling == "gr00t":
            sample = self.beta_dist.sample([batch_size]).to(device=device, dtype=torch.float32)
            return ((1.0 - sample) * self.noise_s).to(dtype=dtype)
        sample = self.beta_dist.sample([batch_size]).to(device=device, dtype=dtype).clamp(max=self.noise_s)
        return (self.noise_s - sample) / self.noise_s

    def _embed_noisy(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 3 or z.shape[-1] != self.d_dino:
            raise ValueError(f"Expected visual target [B,N,{self.d_dino}], got {tuple(z.shape)}")
        if z.shape[1] > self.max_target_tokens:
            raise ValueError(
                f"Visual target has {z.shape[1]} tokens, exceeding max_target_tokens={self.max_target_tokens}. "
                "Configure the bound to the single target image/composite patch grid."
            )
        x = self.x_embed(z)
        if self.add_pos_embed:
            pos = torch.arange(z.shape[1], device=z.device)
            x = x + self.position_embedding(pos).unsqueeze(0)
        return x

    ######### // code // ##########
    def forward(
        self,
        cond: torch.Tensor,
        z_gt: torch.Tensor,
        weights: torch.Tensor | None = None,
        return_pred: bool = False,
        return_details: bool = False,
    ):
        device_type = z_gt.device.type
        autocast_ctx = (
            torch.autocast(device_type=device_type, enabled=False) if device_type in {"cuda", "cpu"} else nullcontext()
        )
        with autocast_ctx:
            #######
            compute_dtype = self._module_dtype(self)
            z_gt = z_gt.to(dtype=compute_dtype)
            cond = cond.to(dtype=compute_dtype)
            noise = torch.randn_like(z_gt)
            t = self.sample_time(z_gt.shape[0], z_gt.device, z_gt.dtype)[:, None, None]
            noisy = (1 - t) * noise + t * z_gt
            velocity = z_gt - noise
            t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()

            hidden = self._embed_noisy(noisy)
            out = self.model(
                hidden_states=hidden,
                encoder_hidden_states=cond,
                timestep=t_discretized,
                return_all_hidden_states=False,
            )
            prediction = self.x_decode(out)
            pred_velocity = visual_prediction_to_velocity(
                prediction,
                noisy,
                t,
                prediction_type=self.prediction_type,
                t_eps=self.jit_t_eps,
            )
            # Preserve the historical velocity objective byte-for-byte for
            # old configs/checkpoints.  Only JiT-x uses the endpoint-derived
            # target, matching GR00T_ActionHeader's parameterization contract.
            target_velocity = velocity
            if self.prediction_type == "jit_x":
                target_velocity = visual_prediction_to_velocity(
                    z_gt,
                    noisy,
                    t,
                    prediction_type="jit_x",
                    t_eps=self.jit_t_eps,
                )
            flow_per_patch = ((pred_velocity.float() - target_velocity.float()) ** 2).mean(dim=-1)

            # A velocity-only objective can have a low scalar loss while still
            # leaving visible Gaussian residue after a short Euler solve.  Add
            # optional direct endpoint supervision in DINO token space.  For
            # jit_x the clean prediction is the network output itself; for the
            # legacy velocity head it is the analytical x1 estimate.
            if self.prediction_type == "jit_x":
                clean_prediction = prediction
            else:
                clean_prediction = noisy + (1.0 - t) * pred_velocity
            clean_per_patch = ((clean_prediction.float() - z_gt.float()) ** 2).mean(dim=-1)
            cosine_per_patch = 1.0 - F.cosine_similarity(
                clean_prediction.float(), z_gt.float(), dim=-1, eps=1.0e-8
            )
            per_patch = (
                flow_per_patch
                + self.clean_target_loss_weight * clean_per_patch
                + self.cosine_loss_weight * cosine_per_patch
            )
            def reduce_patches(values: torch.Tensor) -> torch.Tensor:
                if weights is None:
                    return values.mean()
                return (
                    values
                    * weights.to(dtype=values.dtype, device=values.device)
                ).mean()

            loss = reduce_patches(per_patch)
            details = None
            if return_details:
                # These three terms are intentionally unscaled by their
                # configured coefficients.  W&B can therefore distinguish a
                # genuinely improving clean-DINO predictor from a changing
                # aggregate caused only by loss-weight choices.
                details = {
                    "flow_loss_raw": reduce_patches(flow_per_patch).detach(),
                    "clean_loss_raw": reduce_patches(clean_per_patch).detach(),
                    "cosine_loss_raw": reduce_patches(cosine_per_patch).detach(),
                }
            if return_pred:
                result = (loss, pred_velocity, per_patch)
                return (*result, details) if return_details else result
            if return_details:
                return loss, details
            return loss

    ######### // code // ##########

    #######
    def predict_latent(
        self,
        cond: torch.Tensor,
        n: int,
        encoder_attention_mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        num_inference_timesteps: int | None = None,
    ) -> torch.Tensor:
        batch_size = cond.shape[0]
        compute_dtype = self._module_dtype(self, fallback=cond.dtype)
        cond = cond.to(dtype=compute_dtype)
        z = torch.randn(batch_size, n, self.d_dino, device=cond.device, dtype=compute_dtype, generator=generator)
        num_steps = int(num_inference_timesteps or self.num_inference_timesteps)
        if num_steps <= 0:
            raise ValueError(f"num_inference_timesteps must be positive, got {num_steps}")
        dt = 1.0 / float(num_steps)
        for step in range(num_steps):
            t_cont = step / float(num_steps)
            t_discretized = int(t_cont * self.num_timestep_buckets)
            timestep = torch.full((batch_size,), t_discretized, device=cond.device, dtype=torch.long)
            hidden = self._embed_noisy(z)
            out = self.model(
                hidden_states=hidden,
                encoder_hidden_states=cond,
                timestep=timestep,
                encoder_attention_mask=encoder_attention_mask,
            )
            prediction = self.x_decode(out)
            pred_velocity = visual_prediction_to_velocity(
                prediction,
                z,
                t_cont,
                prediction_type=self.prediction_type,
                t_eps=self.jit_t_eps,
            )
            z = z + dt * pred_velocity
        return z

    #######

    @torch.inference_mode()
    def predict(self, cond: torch.Tensor, n: int) -> torch.Tensor:
        return self.predict_latent(cond, n=n)


######### // code // ##########
