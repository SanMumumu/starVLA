################# RoboDojo 训练：以下四个任务按需提交，不要整段一次性执行
cd /home/users/sen02.wang/workspace/starvla_dev/Robodojo

##################### Qwen3 base
aidi-inf-cli job submit -f mot_base.yaml -q project-ppu-robot-lab-acloud-bj

##################### Qwen3 joint
aidi-inf-cli job submit -f mot_joint.yaml -q project-ppu-robot-lab-acloud-bj

##################### RynnBrain1.1 base（使用 starvla-rynn-ppu:v1.0）
aidi-inf-cli job submit -f rynn_base.yaml -q project-ppu-robot-lab-acloud-bj

##################### RynnBrain1.1 joint（使用 starvla-rynn-ppu:v1.0）
aidi-inf-cli job submit -f rynn_joint.yaml -q project-ppu-robot-lab-acloud-bj

################# robodojo 推理：server/client 分离（参考 robotwin；现在执行这一段）
cd /home/users/sen02.wang/workspace/starvla_dev/RBT
aidi-inf-cli job submit -f job_server.yaml -q project-h20-robot-lab-acloud-langfang
aidi-inf-cli job submit -f job_client_robodojo.yaml -q project-5090-robot-lab-bcloud-bj

##################### 1. H20 server 的 8卡终端：复制执行下面整段
cd /running_package/starvla_dev/starVLA
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export ALBUMENTATIONS_DISABLE_VERSION_CHECK=1
export NO_ALBUMENTATIONS_UPDATE=1
export ROBODOJO_CKPT=/horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/outputs/starvla_robodojo/starvla_qwengroot_robodojo_baseline_h16_jitx_corrnoise_zscore/checkpoints/steps_80000_pytorch_model.pt
export BASE_PORT=7777
export NUM_SERVERS=8
test -f "${ROBODOJO_CKPT}"
# 先记下这个 server job 的内网 IP，client 的 HOST 要填这个 IP。
hostname -I
# 8 张 H20 各加载一个相同 checkpoint，监听 7777..7784；保持终端运行。
bash examples/RoboDojo/eval_files/run_robodojo_policy_servers_8.sh

##################### 2. RoboDojo client 的 8卡终端：替换 HOST 后复制执行下面整段
cd /running_package/starvla_dev/starVLA
export HOST=10.237.116.226 # 必须替换为上面 hostname -I 显示的 H20 server 内网 IP
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export STARVLA_SERVER_HOST="${HOST}"
export STARVLA_SERVER_PORT=7777
export BASE_PORT=7777
export NUM_CLIENTS=8
export ROBODOJO_LOCAL_BASE_PORT=17777
export STARVLA_ROOT="${PWD}"
export ROBODOJO_ROOT=/horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/RoboDojo
export STARVLA_CKPT_PATH=/horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/outputs/starvla_robodojo/starvla_qwengroot_robodojo_baseline_h16_jitx_corrnoise_zscore/checkpoints/steps_80000_pytorch_model.pt
export ROBODOJO_CKPT_NAME=robodojo_baseline_steps80000_h16_state_zscore
export ROBODOJO_TASK=stack_bowls
export ROBODOJO_RUN_NAME=baseline_stack_bowls
export ROBODOJO_POLICY_GPU=0
export ROBODOJO_ENV_GPU=1
# 5090 RoboDojo client 镜像的 Isaac Sim Python；不调用 conda，也不修改共享盘代码。
export ROBODOJO_PYTHON=/opt/robodojo-env/bin/python
export PATH=/opt/robodojo-env/bin:${PATH}
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
export PRIVACY_CONSENT=Y
test -x "${ROBODOJO_PYTHON}"
test -f "${STARVLA_CKPT_PATH}"
# 兼容 websockets 12 且会校验实际 adapter 路径；不要再 sed 共享盘 XPolicyLab。
# 旧统一入口保留；未指定 mode 时现在默认为 fast（不存视频）。
# bash examples/RoboDojo/eval_files/run_aidi_robodojo.sh
# 下面 2A / 2B 二选一执行，不要整段连续执行。

##################### 2A. 快速 rollout：官方全部 trials，不编码/保存视频
# 完整论文协议：42 个任务、54 个 rollout job、2100 episodes。
# Generalization: 12 x (standard 25 + random 25)；其余 30 x 50。
# 8 个 client GPU 动态调度到 H20 的 7777..7784；全程不编码/保存视频。
ROBODOJO_FULL_RUN_NAME=baseline_steps80000_full_official_fast \
bash examples/RoboDojo/eval_files/run_aidi_robodojo_fast_full.sh

##################### 2B. 可视化 rollout：修改 TASK 和 TRIALS，每个 trial 保存多相机 MP4
ROBODOJO_TASK=stack_bowls \
ROBODOJO_TRIALS=5 \
ROBODOJO_RUN_NAME=visualize_stack_bowls_trials5 \
bash examples/RoboDojo/eval_files/run_aidi_robodojo_visualize.sh
