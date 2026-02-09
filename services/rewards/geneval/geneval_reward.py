from __future__ import annotations

from typing import Any

import torch

from services.rewards.base import BaseReward
from services.rewards.geneval.gen_eval import load_geneval
from services.rewards.registry import register_reward


class GenevalReward(BaseReward):
    """
    Local Geneval reward wrapper. Uses vendored gen_eval under rewards/local/geneval.
    """

    def __init__(
        self,
        device: torch.device | str = "cpu",
        config_path: str | None = None,
        ckpt_root: str | None = None,
        object_names_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(device=device, **kwargs)
        self.inference_fn = load_geneval(
            device=str(self.device),
            config_path=config_path,
            ckpt_root=ckpt_root,
            object_names_path=object_names_path,
        )

    def __call__(self, samples):
        sample_list = list(samples)
        images: list[Any] = []
        meta_datas: list[Any] = []
        only_strict: bool | None = None
        for sample in sample_list:
            data = sample if isinstance(sample, dict) else sample.to_dict()
            image = data.get("multimodal_outputs")["image"]
            images.append(image)

            metadata = data.get("metadata")
            if only_strict is None and isinstance(metadata, dict):
                only_strict = bool(metadata.get("only_strict"))
            meta_datas.append(metadata)

        if only_strict is None:
            only_strict = True
        return self.inference_fn(images, meta_datas, only_strict)


@register_reward("geneval")
def build_geneval_reward(
    *, torch_device: torch.device | str | None = None, torch_dtype=None, **kwargs: Any
) -> BaseReward:
    if torch_device is not None and "device" not in kwargs:
        kwargs["device"] = torch_device
    return GenevalReward(**kwargs)
