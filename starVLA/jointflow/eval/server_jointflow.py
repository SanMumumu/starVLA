"""JointFlow LIBERO policy server entry.

所属：JointFlow LIBERO eval（server 端入口）。
复用（import，不改源）：
- starVLA.jointflow.eval.jointflow_server_wrapper.JointFlowPolicyServerWrapper
- deployment.model_server.tools.websocket_policy_server.WebsocketPolicyServer

与 deployment/model_server/server_policy.py 同结构，仅把 wrapper 换成
JointFlow 版（其内部会注册 QwenJointFlow framework、归一化 state、加载 live DINO）。
server 绑定 0.0.0.0，因此跨机时 client 用 server 节点 IP:port 即可连接。

server 发现（rendezvous）由上层 AIDI 入口 aidi/eval/run_server_aidi.sh 负责发布
（调用 starVLA.jointflow.eval.rendezvous publish），本入口只管单个 server 进程。
"""

import argparse
import logging
import socket

from starVLA.jointflow.eval.jointflow_server_wrapper import JointFlowPolicyServerWrapper
from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer


######### // code // ##########
# 中文注释：构建 JointFlow wrapper 并启动 websocket server。
# wrapper 负责：注册 framework / 加载 ckpt / 动作反归一化 / state 归一化 / live DINO。
def main(args) -> None:
    wrapper = JointFlowPolicyServerWrapper(
        ckpt_path=args.ckpt_path,
        device="cuda",
        use_bf16=args.use_bf16,
        unnorm_key=args.unnorm_key,
    )

    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except Exception:
        local_ip = "0.0.0.0"
    logging.info("JointFlow server (host=%s ip=%s port=%d) metadata=%s", hostname, local_ip, args.port, wrapper.metadata)

    server = WebsocketPolicyServer(
        policy=wrapper,
        host="0.0.0.0",
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata=wrapper.metadata,
    )
    server.serve_forever()
######### // code // ##########


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to <run_dir>/checkpoints/x.pt")
    parser.add_argument("--port", type=int, default=6500)
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--unnorm_key", type=str, default=None)
    parser.add_argument("--idle_timeout", type=int, default=-1, help="Idle timeout seconds; -1 = never close")
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    args = build_argparser().parse_args()
    main(args)
