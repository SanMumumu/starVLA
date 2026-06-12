"""训练指标 tracker 适配层：默认 SwanLab，JOINTFLOW_LOGGER=wandb 可切回。

为什么：wandb 上行网络不稳（用户反馈），换 SwanLab；trainer 侧调用面保持 wandb 习惯
（init/log/finish/run.summary），本模块做参数名映射，train_jointflow.py 只改一行 import。

映射（swanlab 0.8.x 的 init 形参与 wandb 几乎同名）：
  name→name, project→project, group→group, dir→log_dir, mode→mode("online/offline/disabled" 同义),
  id→id, resume("allow")→resume=True；entity 丢弃（swanlab 的 workspace 默认当前账号）。
登录：online 模式下用 env SWANLAB_API_KEY 调 swanlab.login（失败仅告警并降级 offline，不阻塞训练）。
log 只透传有限数值标量（wandb 允许 str/None，swanlab 不保证 → 过滤）。
"""

from __future__ import annotations

import math
import os


######### // code // ##########
class _Run:
    # 中文注释：兼容 wandb.run.summary[...] = v 的最小面（仅存内存，不上传——参数表 stdout 已打印）。
    def __init__(self):
        self.summary: dict = {}


class _SwanlabTracker:
    def __init__(self):
        self._sw = None
        self.run = None

    def init(self, **kwargs):
        import swanlab

        self._sw = swanlab
        mode = str(kwargs.get("mode", "online")).strip().lower()
        if mode not in {"online", "offline", "local", "disabled"}:
            mode = "online"
        if mode == "online":
            api_key = os.environ.get("SWANLAB_API_KEY", "").strip()
            if api_key:
                try:
                    swanlab.login(api_key=api_key, save=True)
                except Exception as exc:  # noqa: BLE001
                    print(f"[track_logger][warn] swanlab.login failed ({exc}); fallback mode=offline", flush=True)
                    mode = "offline"
        init_kwargs = {
            "project": kwargs.get("project"),
            "name": kwargs.get("name"),
            "group": kwargs.get("group"),
            "log_dir": kwargs.get("dir"),
            "mode": mode,
        }
        # 中文注释（坑）：swanlab 规定 id 必须配 resume!=never（否则 assert
        # "Run id should not be provided when resume=never."）。因此：
        #   - 全新 run **不传 id**（实验名已=RUN_NAME，面板可检索，无信息损失）；
        #   - 续训才 id+resume="allow" 一起传（x 轴接上同一实验）。
        if kwargs.get("resume") and kwargs.get("id"):
            init_kwargs["id"] = str(kwargs["id"])
            init_kwargs["resume"] = "allow"
        try:
            self._sw.init(**{k: v for k, v in init_kwargs.items() if v is not None})
        except (TypeError, AssertionError, ValueError) as exc:
            # 中文注释：swanlab 版本差异（不认 id/resume/group 形参，或 id 格式校验不过）→
            # 去掉这些可选项降级重试；续训曲线会另起一条，但训练不被日志问题阻塞。
            print(f"[track_logger][warn] swanlab.init degraded ({exc})", flush=True)
            for key in ("id", "resume", "group"):
                init_kwargs.pop(key, None)
            self._sw.init(**{k: v for k, v in init_kwargs.items() if v is not None})
        self.run = _Run()

    def log(self, metrics: dict, step: int | None = None):
        if self._sw is None:
            return
        clean = {}
        for key, value in metrics.items():
            if isinstance(value, bool):
                value = int(value)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                clean[key] = value
        if clean:
            self._sw.log(clean, step=step)

    def finish(self):
        if self._sw is not None:
            self._sw.finish()


def get_tracker():
    backend = os.environ.get("JOINTFLOW_LOGGER", "swanlab").strip().lower()
    if backend == "wandb":
        import wandb

        return wandb
    return _SwanlabTracker()
######### // code // ##########
