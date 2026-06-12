"""PR7: TriFlow LIBERO policy server 入口.

复用（import，不改源）:
- starVLA.triflow.eval.triflow_server_wrapper.TriFlowPolicyServerWrapper
- deployment.model_server.tools.websocket_policy_server.WebsocketPolicyServer

与 v1 server_jointflow.py 同结构；client 用 v1 的
starVLA/jointflow/eval/eval_libero_jointflow.py 原样连接（它发的 state 被忽略）。
"""

import argparse
import logging
import socket

from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
from starVLA.triflow.eval.triflow_server_wrapper import TriFlowPolicyServerWrapper


######### // code // ##########
# 中文注释：构建 TriFlow wrapper 并启动 websocket server（fp32；动作反归一化在父类）。
def main(args) -> None:
    wrapper = TriFlowPolicyServerWrapper(
        ckpt_path=args.ckpt_path,
        device="cuda",
        use_bf16=False,
        unnorm_key=args.unnorm_key,
    )
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except Exception:
        local_ip = "0.0.0.0"
    logging.info("TriFlow server (host=%s ip=%s port=%d) metadata=%s", hostname, local_ip, args.port, wrapper.metadata)
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
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to <run_dir>/checkpoints/x.pt (建议用 *_ema.pt)")
    parser.add_argument("--port", type=int, default=6500)
    parser.add_argument("--unnorm_key", type=str, default=None)
    parser.add_argument("--idle_timeout", type=int, default=-1)
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(build_argparser().parse_args())
