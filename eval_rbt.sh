# baseline 复现，检查训练代码是否正确？
aidi-inf-cli job submit -f robotwin_qwengroot.yaml -q project-h20-horizon-labs-acloud
aidi-inf-cli job submit -f robotwin_qwenoft.yaml -q project-h20-horizon-labs-acloud
# rand2clean (27500 条数据)
aidi-inf-cli job submit -f robotwin_wam_warmup_rand.yaml -q project-h20-horizon-labs-acloud
aidi-inf-cli job submit -f robotwin_wam_gate_rand2clean.yaml -q project-h20-horizon-labs-acloud
# clean2clean (steps可以更小，因为数据量只有2500条) 
aidi-inf-cli job submit -f robotwin_wam_warmup_clean.yaml -q project-h20-horizon-labs-acloud
aidi-inf-cli job submit -f robotwin_wam_gate_clean2clean.yaml -q project-h20-horizon-labs-acloud
#################

################# 测评 robotwin 8卡（debug 全套验证）
# 说明：请先在两台服务器上分别启动 server / client job，保持它们在线；然后在任意可访问终端执行下面命令。
# 本 debug 版本会：
#   1) 检查 checkpoint 是否完整加载（missing / unexpected keys）
#   2) 检查 RoboTwin server 输入/输出动作是否正常
#   3) 以小规模 episodes 运行 8 卡评测
#   4) 输出保存到 examples/Robotwin/eval_debug/outputs/<run_name>_debug_*/
#
# 三条待对照路径：
#   A. official ckpt eval
#   B. qwenOFT-1epoch(frame sample)
#   C. WAM warmup + gate finetune
#
# 你只需要改下面的 CKPT/HOST，然后运行这条 debug 命令。
#
# 示例：
# bash examples/Robotwin/eval_debug/run_robotwin_eval_debug.sh \
#   CKPT="/horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/outputs/starvla_wam_robotwin/robotwin_wam_m5_gate_rand2clean_h16/checkpoints/steps_25000_pytorch_model.pt" \
#   HOST="172.16.213.42" \
#   BASE_PORT=6698 \
#   NUM_CLIENTS=8 \
#   MODES="demo_clean demo_randomized" \
#   TASKS=all \
#   RUN_NAME=robotwin_debug \
#   ROBOTWIN_TEST_NUM=5 \
#   ROBOTWIN_PATH="/horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/RoboTwin"
#
# 单独检索：
# python examples/Robotwin/eval_debug/check_robotwin_ckpt_load.py --config examples/Robotwin/train_files/robotwin_wam_gate_rand2clean.yaml --ckpt "$CKPT"
# python examples/Robotwin/eval_debug/check_robotwin_eval_input.py --ckpt "$CKPT" --host 172.16.213.42 --port 6698

# ========================== 下面是正式评测入口 ==========================
aidi-inf-cli job submit -f job_server.yaml -q project-h20-horizon-labs-acloud

cd /running_package/starvla_dev/starVLA
CKPT=/horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/outputs/starvla_robotwin/Qwen3-VL-OFT-RoboTwin2-All/checkpoints/steps_140000_pytorch_model.pt \
BASE_PORT=6698 \
NUM_SERVERS=8 \
bash examples/Robotwin/eval_files/run_policy_servers_8.sh

aidi-inf-cli job submit -f job_client_robotwin.yaml -q project-h20-horizon-labs-acloud

cd /running_package/starvla_dev/starVLA
export ROBOTWIN_PATH=/horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/RoboTwin
export PYTHONPATH="${ROBOTWIN_PATH}:${PYTHONPATH:-}"
CKPT=/horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/outputs/starvla_wam_robotwin/robotwin_wam_m5_gate_rand2clean_h16/checkpoints/steps_25000_pytorch_model.pt \
HOST=172.16.213.42 \
BASE_PORT=6698 \
NUM_CLIENTS=8 \
MODES="demo_clean demo_randomized" \
TASKS=all \
RUN_NAME=epoch_1 \
ROBOTWIN_TEST_NUM=20 \
ROBOTWIN_PATH="${ROBOTWIN_PATH}" \
bash examples/Robotwin/eval_files/eval_robotwin_8clients_fast.sh
