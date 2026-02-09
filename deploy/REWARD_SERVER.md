Reward Server 说明

目标
- 把评测/打分逻辑从本地推理中拆出来，独立部署成服务。
- 通过 Ray Serve 启动（`deploy/deploy.py`），支持批处理与多副本扩展。
- 与 `gym_v.envs.eval.t2ieval` 的 Sample JSON 请求对齐。

架构
```
Env/Client
  -> RewardClient (POST /v1/generate)
  -> Reward Server (Ray Serve)
  -> RMModelManager (services/rewards)
  -> Reward 实现 (geneval / qwen3vl / ...)
```
说明：
- `deploy/server.py` 接收 Sample JSON，解码图片/视频字段后转为 `Sample`。
- `RMModelManager` 根据 `SCORE_JSON` 构建 reward 组件，逐个计算并返回字典结果。

文件结构
- `deploy/server.py`：Ray Serve 服务实现（Sample JSON）。
- `deploy/deploy.py`：部署入口。
- `services/rewards/`：Reward 实现与模型依赖。
- `start.sh`：最简启动脚本（默认 geneval）。

接口路径
- `POST /v1/generate`：Sample JSON 协议。

启动方式
```
./start.sh
```

环境变量
- `SCORE_JSON`：reward 配置，默认 `{"geneval":{...}}`（见 `start.sh`）。
- `DEVICE`：cuda/cpu（默认 cuda）。
- `PORT`：默认 18085。
- `NUM_GPUS`：每个实例占用的 GPU 数量（默认 1，可设为 >1）。
- `NUM_REPLICAS`：可选，指定实例数；不设置则根据 Ray 集群 GPU 总数和 `NUM_GPUS` 自动计算。
- `RANK`/`NODE_RANK`：多节点时用于区分 head/worker，rank=0 为 head（torchrun 会自动设置 `RANK`）。
- `MASTER_ADDR`/`RAY_HEAD_ADDR`：worker 连接 head 的地址（torchrun 会自动设置 `MASTER_ADDR`）。
- `RAY_PORT`：Ray 端口，默认 6379。
- `WORKER_HOLD`：worker 启动后是否阻塞等待（默认 0，设为 1 则 `tail -f /dev/null`）。
- `MAX_ONGOING_REQUESTS`：可选，Ray Serve 的 `max_ongoing_requests`（start.sh 透传）。

启动参数（deploy.py）
- `--max-ongoing-requests`：Ray Serve 单副本最大并发请求数（默认 5120）。

示例（传入 Geneval 模型路径）
```
SCORE_JSON='{"geneval":{"torch_device":"cuda","config_path":"...","ckpt_root":"...","object_names_path":"..."}}' ./start.sh
```

示例（多 reward）
```
SCORE_JSON='{
  "geneval": {"torch_device":"cuda","config_path":"...","ckpt_root":"..."},
  "qwen3vl": {"model":"Qwen/Qwen3-VL-8B-Instruct","device":"cuda"}
}' ./start.sh
```
说明：
- 支持为同一 reward 传入 list-of-dicts，例如 `{"qwen3vl": [{...}, {...}]}`，会注册为 `qwen3vl_0/qwen3vl_1`。
- `qwen3vl` 需要在请求的 `metadata.gen` 中提供 `max_tokens/temperature/logprobs`。

新增 reward
- 新建文件：`services/rewards/<reward_name>/<reward_name>_reward.py`（文件名需以 `_reward.py` 结尾，才能被自动扫描注册）。
- 继承 `BaseReward` 并实现 `__call__(self, samples)`。
- 用 `@register_reward("<reward_name>")` 注册 builder（工厂函数）。
- 通过 `SCORE_JSON` 传入该 reward 的初始化参数。

示例（最小骨架）
```
from services.rewards.base import BaseReward
from services.rewards.registry import register_reward

class MyReward(BaseReward):
    def __init__(self, device="cpu", **kwargs):
        super().__init__(device=device)
        # 初始化模型/状态

    def __call__(self, samples):
        results = []
        for sample in samples or []:
            data = sample if isinstance(sample, dict) else sample.to_dict()
            # 计算分数
            results.append({"score": 0.0})
        return results

@register_reward("myreward")
def build_myreward(*, torch_device=None, **kwargs):
    if torch_device is not None and "device" not in kwargs:
        kwargs["device"] = torch_device
    return MyReward(**kwargs)
```

Reward 配置结构（SCORE_JSON）
- 顶层是 dict：`{ "<reward_name>": <config> }`
- `<config>` 支持 dict 或 list[dict]：
  - dict：单实例。
  - list[dict]：多实例，会注册为 `<reward_name>_0/<reward_name>_1/...`。

Reward 返回结构
- reward 的 `__call__` 可以返回：
  - `list`：长度应与样本数一致（推荐）。
  - `dict` 或单值：服务端会按样本数复制。
- 最终单请求返回一个 dict，key 为 reward 名称；批量请求时，每个请求对应一个 dict。

Sample JSON 字段解析
- `prompt`：str 或消息列表（list[dict]）。
- `multimodal_inputs`：输入多模态（如图像/视频），用于模型理解。
- `multimodal_outputs`：输出多模态（如生成图像），常用于 T2I 评分。
- `multimodal_train_inputs`：训练态输入（可选）。
- `response`：文本输出（可选）。
- `metadata` / `train_metadata`：额外配置/控制参数（如 `metadata.gen`）。
- 其他未列字段会被保留并透传给 reward。

多模态字段解码规则
- 仅解码 `multimodal_inputs` / `multimodal_outputs` / `multimodal_train_inputs` 下的：
  - `image` / `images`
  - `video` / `videos`
- 支持 data URL、本地路径，以及 list 形式（不支持 raw base64/HTTP(S) URL）。

请求格式（Sample JSON，单图示例）
```
curl -s http://127.0.0.1:18085/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "geneval",
    "prompt": "score this image",
    "multimodal_outputs": {"image": "data:image/jpeg;base64,..."},
    "metadata": {
      "...": "...",
      "only_strict": true
    }
  }'
```

说明
- 服务端不解析 prompt/images/metadata，仅对 Sample 内部的 image/video 字段做 data URL 或本地路径解码。
- 解码后的 Sample（dict）会直接传给 reward 实现。

返回格式
- 成功：直接返回 reward 输出 dict（key 为 reward 名称）。
- 失败：reward 返回 `{"error": "..."}` 时为 HTTP 400；运行时异常会由 FastAPI/Ray 直接返回 500。
