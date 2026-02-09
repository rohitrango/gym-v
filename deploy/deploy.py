from __future__ import annotations

import argparse
import json
import math
import threading
from typing import Any

import ray
from ray import serve

from deploy.server import build_reward_service


def _parse_mapping(text: str) -> dict[str, Any]:
    return json.loads(text)


def _load_score_dict(score_json: str | None, score_path: str | None) -> dict[str, Any]:
    if score_path:
        with open(score_path, encoding="utf-8") as handle:
            return _parse_mapping(handle.read())
    if score_json:
        return _parse_mapping(score_json)
    return {}


def _default_device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _cluster_gpu_total() -> float:
    resources = ray.cluster_resources()
    return float(resources.get("GPU", 0.0))


def _resolve_num_replicas(
    num_replicas: int | None,
    num_gpus: float,
    *,
    total_gpus: float,
) -> int:
    if num_replicas is not None:
        return num_replicas
    if num_gpus <= 0 or total_gpus <= 0:
        return 1
    replicas = int(math.floor(total_gpus / num_gpus))
    return replicas


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deploy gym-v reward server with Ray Serve."
    )
    parser.add_argument(
        "--score-json", help="Reward spec as JSON or Python dict string."
    )
    parser.add_argument(
        "--score-path", help="Path to reward spec JSON or Python dict file."
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Reward device, e.g. cuda or cpu. Defaults to cuda if the Ray cluster has GPUs.",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18085)
    parser.add_argument("--route-prefix", default="/")
    parser.add_argument("--max-batch-size", type=int, default=32)
    parser.add_argument("--batch-wait-s", type=float, default=0.01)
    parser.add_argument(
        "--max-ongoing-requests",
        type=int,
        default=5120,
        help="Ray Serve max_ongoing_requests per replica.",
    )
    parser.add_argument(
        "--num-replicas",
        type=int,
        default=None,
        help="Replica count. If omitted, auto-calculate from total GPUs and num-gpus.",
    )
    parser.add_argument(
        "--num-gpus",
        type=float,
        default=None,
        help="GPUs per replica. Defaults to 1 for cuda and 0 for cpu.",
    )
    parser.add_argument("--num-cpus", type=float, default=None)
    parser.add_argument(
        "--ray-address",
        default=None,
        help="Ray address, e.g. auto or ray://<ip>:10001.",
    )
    parser.add_argument("--name", default="gym_v_reward")
    args = parser.parse_args()

    score_dict = _load_score_dict(args.score_json, args.score_path)

    if args.ray_address:
        ray.init(address=args.ray_address)
    else:
        ray.init()

    cluster_gpus = _cluster_gpu_total()
    device = args.device
    if device is None:
        device = "cuda" if cluster_gpus > 0 else _default_device()

    num_gpus = args.num_gpus
    if num_gpus is None:
        num_gpus = 1.0 if "cuda" in device else 0.0

    num_replicas = _resolve_num_replicas(
        args.num_replicas,
        num_gpus,
        total_gpus=cluster_gpus,
    )

    serve.start(http_options={"host": args.host, "port": args.port})

    RewardService = build_reward_service(
        max_batch_size=args.max_batch_size,
        batch_wait_timeout_s=args.batch_wait_s,
        max_ongoing_requests=args.max_ongoing_requests,
    )
    ray_actor_options = {"num_gpus": num_gpus}
    if args.num_cpus is not None:
        ray_actor_options["num_cpus"] = args.num_cpus

    app = RewardService.options(
        num_replicas=num_replicas,
        ray_actor_options=ray_actor_options,
    ).bind(score_dict=score_dict, device=device)

    serve.run(app, name=args.name, route_prefix=args.route_prefix)
    threading.Event().wait()


if __name__ == "__main__":
    main()
