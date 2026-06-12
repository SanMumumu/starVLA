"""JointFlow eval rendezvous helpers (multi-node server discovery).

所属：JointFlow LIBERO eval（server/client 拆分拓扑）。
复用：仅标准库 + socket，不依赖外部服务。
为何不改源文件：这是 JointFlow 评测新增的多机协调逻辑，starVLA 既有 eval
脚本（examples/LIBERO/.../auto_eval_scripts）只支持单机轮询，无法表达
"A 机起 server / B 机起 client" 的跨机发现，因此在 jointflow 内新写。

设计：用户已确认采用"共享 bucket 上的 rendezvous 文件"做发现。
server 节点把自己的可路由 IP + 各 server 端口写入 <rdv_dir>/servers.json，
client 节点轮询该文件直到就绪后再连接。run_root_dir 在挂载的 /horizon-bucket
上，两端都能读写，因此无需知道 AIDI 具体的 peer-IP 环境变量。
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import time
from pathlib import Path
from typing import List, Optional


SERVERS_FILENAME = "servers.json"


######### // code // ##########
# 中文注释：获取本节点的可路由 IP（供 client 跨机连接 server 用）。
# 优先用 `hostname -I` 的第一个地址（集群里通常是内网可达地址）；
# 失败再退回 socket 连一个外部地址来推断本机出口 IP；最后退回 hostname 解析。
def get_node_ip() -> str:
    env_ip = os.environ.get("JOINTFLOW_NODE_IP", "").strip()
    if env_ip:
        return env_ip

    try:
        out = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=5)
        addrs = out.stdout.split()
        if addrs:
            return addrs[0]
    except Exception:
        pass

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
        finally:
            sock.close()
    except Exception:
        pass

    try:
        return socket.gethostbyname(socket.gethostname())
    except Exception:
        return "127.0.0.1"
######### // code // ##########


######### // code // ##########
# 中文注释：server 节点发布自己的 host + 端口列表到 <rdv_dir>/servers.json。
# 采用"先写临时文件再原子 rename"，避免 client 读到半截 JSON。
# servers 结构：[{"host": ip, "port": p}, ...]，顺序即 client 分配顺序。
def publish_servers(rdv_dir: str | Path, host: str, ports: List[int]) -> Path:
    rdv_dir = Path(rdv_dir)
    rdv_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "ready": True,
        "host": host,
        "timestamp": time.time(),
        "servers": [{"host": host, "port": int(p)} for p in ports],
    }
    final_path = rdv_dir / SERVERS_FILENAME
    tmp_path = rdv_dir / f"{SERVERS_FILENAME}.tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, final_path)
    return final_path
######### // code // ##########


######### // code // ##########
# 中文注释：client 节点轮询 <rdv_dir>/servers.json，直到 ready 且 server 数 >= min_servers。
# 返回 [{"host","port"}, ...]；超时抛 TimeoutError。poll_interval 秒轮询一次。
def wait_for_servers(
    rdv_dir: str | Path,
    min_servers: int = 1,
    timeout: float = 1800.0,
    poll_interval: float = 5.0,
) -> List[dict]:
    rdv_dir = Path(rdv_dir)
    path = rdv_dir / SERVERS_FILENAME
    start = time.time()
    while True:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                servers = payload.get("servers", []) if payload.get("ready") else []
                if len(servers) >= min_servers:
                    return servers
            except (json.JSONDecodeError, OSError):
                pass  # 中文注释：文件正在被写、读到半截，下一轮重试。
        if time.time() - start > timeout:
            raise TimeoutError(
                f"wait_for_servers timeout after {timeout}s; expected >= {min_servers} servers at {path}"
            )
        time.sleep(poll_interval)
######### // code // ##########


######### // code // ##########
# 中文注释：解析 client 应连接的 server。优先用显式 env（JOINTFLOW_SERVER_HOST），
# 否则从 rendezvous 文件取。host 为 None 时表示本机（localhost）单机模式。
def resolve_server_for_client(
    client_index: int,
    num_servers: int,
    rdv_dir: Optional[str | Path] = None,
    explicit_host: Optional[str] = None,
    base_port: int = 6500,
    wait_timeout: float = 1800.0,
) -> dict:
    if explicit_host:
        port = base_port + (client_index % max(num_servers, 1))
        return {"host": explicit_host, "port": port}
    if rdv_dir is not None:
        servers = wait_for_servers(rdv_dir, min_servers=1, timeout=wait_timeout)
        return servers[client_index % len(servers)]
    # 中文注释：单机模式，server 与 client 同机，连 localhost。
    port = base_port + (client_index % max(num_servers, 1))
    return {"host": "127.0.0.1", "port": port}
######### // code // ##########


######### // code // ##########
# 中文注释：命令行入口，供 aidi/eval/run_*_aidi.sh 调用。
# - ip                                  打印本节点可路由 IP。
# - publish --rdv_dir D --base_port P --num_servers N [--host H]
#                                       发布 servers.json（host 默认本机 IP，端口 P..P+N-1）。
# - wait --rdv_dir D --min_servers M [--timeout T]
#                                       轮询到就绪后，每行打印 "host port"（供 bash 读成数组）。
def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ip")

    p_pub = sub.add_parser("publish")
    p_pub.add_argument("--rdv_dir", required=True)
    p_pub.add_argument("--base_port", type=int, required=True)
    p_pub.add_argument("--num_servers", type=int, required=True)
    p_pub.add_argument("--host", default=None)

    p_wait = sub.add_parser("wait")
    p_wait.add_argument("--rdv_dir", required=True)
    p_wait.add_argument("--min_servers", type=int, default=1)
    p_wait.add_argument("--timeout", type=float, default=1800.0)

    args = parser.parse_args()
    if args.cmd == "ip":
        print(get_node_ip())
    elif args.cmd == "publish":
        host = args.host or get_node_ip()
        ports = [args.base_port + i for i in range(args.num_servers)]
        path = publish_servers(args.rdv_dir, host, ports)
        print(f"published {len(ports)} servers (host={host}) -> {path}")
    elif args.cmd == "wait":
        servers = wait_for_servers(args.rdv_dir, min_servers=args.min_servers, timeout=args.timeout)
        for s in servers:
            print(f"{s['host']} {s['port']}")
######### // code // ##########


if __name__ == "__main__":
    _main()
