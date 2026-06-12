######### // code // ##########
# 中文注释：e101 —— LIBERO object 高分 recipe 回归（在线 DINO 改造后的基线）。
# 目的：online-DINO 切换 / 框架近期改动后，确认 LIBERO SR 不回退（对照历史离线 recipe 结果）。
# 评测：aidi/eval/ 的 server+client 双任务对（TASK_SUITES=libero_object, 50 trials/task）。
######### // code // ##########
CONFIG=starVLA/jointflow/configs/jointflow_libero_highscore_b24.yaml
RUN_NAME=jf_e101_libero_object_highscore
# 中文注释：e101 有意保持在线 DINO（这是它要回归的对象），但集群无外网 →
# repo/权重必须指 bucket 本地路径，否则 lazy 加载 DINO 时 torch.hub/HF 联网直接崩。
OVERRIDES=(
  --framework.dino.load_live_backbone=true
  --framework.dino.repo_or_dir=/horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/CKPTS/dinov3_repo
  --framework.dino.weights=/horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/CKPTS/dinov3-vits16-pretrain-lvd1689m/
)
