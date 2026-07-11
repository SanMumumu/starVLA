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
aidi-inf-cli job submit -f job_server.yaml -q project-h20-horizon-labs-acloud

################# 测评 robotwin 8卡
# baseline: /horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/outputs/starvla_robotwin/Qwen3-VL-OFT-RoboTwin2-All/checkpoints/steps_140000_pytorch_model.pt
# Groot: /horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/outputs/starvla_robotwin/starvla_qwengroot_robotwin/checkpoints/steps_60000_pytorch_model.pt
# oft: /horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/outputs/starvla_robotwin/starvla_qwenoft_robotwin/checkpoints/steps_50000_pytorch_model.pt
aidi-inf-cli job submit -f job.yaml -q project-h20-horizon-labs-acloud

cd /running_package/starvla_dev/starVLA
CKPT=/horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/outputs/starvla_robotwin/starvla_qwenoft_robotwin/checkpoints/steps_50000_pytorch_model.pt \
BASE_PORT=6698 \
NUM_SERVERS=8 \
bash examples/Robotwin/eval_files/run_policy_servers_8.sh

aidi-inf-cli job submit -f job_client_robotwin.yaml -q project-h20-horizon-labs-acloud

cd /running_package/starvla_dev/starVLA
export ROBOTWIN_PATH=/horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/RoboTwin
export PYTHONPATH="${ROBOTWIN_PATH}:${PYTHONPATH:-}"
CKPT=/horizon-bucket/robot_lab/users/sen.wang-labs/starVLA/outputs/starvla_robotwin/starvla_qwenoft_robotwin/checkpoints/steps_50000_pytorch_model.pt \
HOST=172.16.213.42 \
BASE_PORT=6698 \
NUM_CLIENTS=8 \
MODES="demo_clean demo_randomized" \
TASKS=all \
RUN_NAME=epoch_1 \
ROBOTWIN_TEST_NUM=100 \
ROBOTWIN_PATH="${ROBOTWIN_PATH}" \
bash examples/Robotwin/eval_files/eval_robotwin_8clients_fast.sh