import json
import os
from pathlib import Path
import sys  # noqa: F401
import time
import warnings

from clip_benchmark.metrics import zeroshot_classification as zsc
import mmdet
from mmdet.apis import inference_detector, init_detector
import numpy as np
import open_clip
from PIL import Image, ImageOps
import torch

warnings.filterwarnings("ignore")
zsc.tqdm = lambda it, *args, **kwargs: it

DEFAULT_CONFIG_PATH = (
    "your_mmdetection_path/mmdetection/configs/mask2former/"
    "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py"
)
DEFAULT_CKPT_ROOT = "your_reward-server_path/reward-server/model/mask2former2"


def load_geneval(
    *,
    device: str | None = None,
    config_path: str | None = None,
    ckpt_root: str | None = None,
    object_names_path: str | None = None,
):
    device = device or os.environ.get("GENEVAL_DEVICE", "cuda")
    config_path = config_path or os.environ.get(
        "GENEVAL_CONFIG_PATH", DEFAULT_CONFIG_PATH
    )
    ckpt_root = ckpt_root or os.environ.get("GENEVAL_CKPT_ROOT", DEFAULT_CKPT_ROOT)
    object_names_path = object_names_path or os.environ.get(
        "GENEVAL_OBJECT_NAMES_PATH",
        str(Path(__file__).with_name("object_names.txt")),
    )

    def timed(fn):
        def wrapper(*args, **kwargs):
            startt = time.time()
            result = fn(*args, **kwargs)
            endt = time.time()
            print(
                f"Function {fn.__name__!r} executed in {endt - startt:.3f}s",
                file=sys.stderr,
            )
            return result

        return wrapper

    # Load models

    @timed
    def load_models():
        config_candidate = config_path
        if not os.path.isabs(config_candidate):
            config_candidate = os.path.join(
                os.path.dirname(mmdet.__file__),
                config_candidate,
            )
        if not os.path.isfile(config_candidate):
            raise FileNotFoundError(
                f"mmdet config not found: {config_candidate}. "
                "Set GENEVAL_CONFIG_PATH or --config-path."
            )
        CONFIG_PATH = config_candidate
        OBJECT_DETECTOR = "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco"
        CKPT_PATH = os.path.join(ckpt_root, f"{OBJECT_DETECTOR}.pth")
        if not os.path.isfile(CKPT_PATH):
            raise FileNotFoundError(
                f"checkpoint not found: {CKPT_PATH}. "
                "Set GENEVAL_CKPT_ROOT or --ckpt-root."
            )
        object_detector = init_detector(CONFIG_PATH, CKPT_PATH, device=device)

        clip_arch = "ViT-L-14"
        clip_model, _, transform = open_clip.create_model_and_transforms(
            clip_arch, pretrained="openai", device=device
        )
        tokenizer = open_clip.get_tokenizer(clip_arch)

        if not os.path.isfile(object_names_path):
            raise FileNotFoundError(
                f"object_names.txt not found: {object_names_path}. "
                "Set GENEVAL_OBJECT_NAMES_PATH or --object-names-path."
            )
        with open(object_names_path) as cls_file:
            classnames = [line.strip() for line in cls_file]

        return object_detector, (clip_model, transform, tokenizer), classnames

    COLORS = [
        "red",
        "orange",
        "yellow",
        "green",
        "blue",
        "purple",
        "pink",
        "brown",
        "black",
        "white",
    ]
    COLOR_CLASSIFIERS = {}

    # Evaluation parts

    class ImageCrops(torch.utils.data.Dataset):
        def __init__(self, image: Image.Image, objects):
            self._image = image.convert("RGB")
            bgcolor = "#999"
            if bgcolor == "original":
                self._blank = self._image.copy()
            else:
                self._blank = Image.new("RGB", image.size, color=bgcolor)
            self._objects = objects

        def __len__(self):
            return len(self._objects)

        def __getitem__(self, index):
            box, mask = self._objects[index]
            if mask is not None:
                assert tuple(self._image.size[::-1]) == tuple(mask.shape), (
                    index,
                    self._image.size[::-1],
                    mask.shape,
                )
                image = Image.composite(self._image, self._blank, Image.fromarray(mask))
            else:
                image = self._image
            image = image.crop(box[:4])
            return (transform(image), 0)

    def color_classification(image, bboxes, classname):
        if classname not in COLOR_CLASSIFIERS:
            COLOR_CLASSIFIERS[classname] = zsc.zero_shot_classifier(
                clip_model,
                tokenizer,
                COLORS,
                [
                    f"a photo of a {{c}} {classname}",
                    f"a photo of a {{c}}-colored {classname}",
                    "a photo of a {c} object",
                ],
                device,
            )
        clf = COLOR_CLASSIFIERS[classname]
        dataloader = torch.utils.data.DataLoader(
            ImageCrops(image, bboxes), batch_size=16, num_workers=4
        )
        with torch.no_grad():
            pred, _ = zsc.run_classification(clip_model, clf, dataloader, device)
            return [COLORS[index.item()] for index in pred.argmax(1)]

    def compute_iou(box_a, box_b):
        def area_fn(box):
            return max(box[2] - box[0] + 1, 0) * max(box[3] - box[1] + 1, 0)

        i_area = area_fn(
            [
                max(box_a[0], box_b[0]),
                max(box_a[1], box_b[1]),
                min(box_a[2], box_b[2]),
                min(box_a[3], box_b[3]),
            ]
        )
        u_area = area_fn(box_a) + area_fn(box_b) - i_area
        return i_area / u_area if u_area else 0

    def relative_position(obj_a, obj_b):
        """Give position of A relative to B, factoring in object dimensions"""
        boxes = np.array([obj_a[0], obj_b[0]])[:, :4].reshape(2, 2, 2)
        center_a, center_b = boxes.mean(axis=-2)
        dim_a, dim_b = np.abs(np.diff(boxes, axis=-2))[..., 0, :]
        offset = center_a - center_b
        #
        revised_offset = np.maximum(
            np.abs(offset) - POSITION_THRESHOLD * (dim_a + dim_b), 0
        ) * np.sign(offset)
        if np.all(np.abs(revised_offset) < 1e-3):
            return set()
        #
        dx, dy = revised_offset / np.linalg.norm(offset)
        relations = set()
        if dx < -0.5:
            relations.add("left of")
        if dx > 0.5:
            relations.add("right of")
        if dy < -0.5:
            relations.add("above")
        if dy > 0.5:
            relations.add("below")
        return relations

    def evaluate(image, objects, metadata):
        """
        Evaluate given image using detected objects on the global metadata specifications.
        Assumptions:
        * Metadata combines 'include' clauses with AND, and 'exclude' clauses with OR
        * All clauses are independent, i.e., duplicating a clause has no effect on the correctness
        * CHANGED: Color and position will only be evaluated on the most confidently predicted objects;
            therefore, objects are expected to appear in sorted order
        """
        correct = True
        reason = []
        matched_groups = []
        # Check for expected objects
        for req in metadata.get("include", []):
            classname = req["class"]
            matched = True
            found_objects = objects.get(classname, [])[: req["count"]]
            if len(found_objects) < req["count"]:
                correct = matched = False
                reason.append(
                    f"expected {classname}>={req['count']}, found {len(found_objects)}"
                )
            else:
                if "color" in req:
                    # Color check
                    colors = color_classification(image, found_objects, classname)
                    if colors.count(req["color"]) < req["count"]:
                        correct = matched = False
                        reason.append(
                            f"expected {req['color']} {classname}>={req['count']}, found "
                            + f"{colors.count(req['color'])} {req['color']}; and "
                            + ", ".join(
                                f"{colors.count(c)} {c}" for c in COLORS if c in colors
                            )
                        )
                if "position" in req and matched:
                    # Relative position check
                    expected_rel, target_group = req["position"]
                    if matched_groups[target_group] is None:
                        correct = matched = False
                        reason.append(f"no target for {classname} to be {expected_rel}")
                    else:
                        for obj in found_objects:
                            for target_obj in matched_groups[target_group]:
                                true_rels = relative_position(obj, target_obj)
                                if expected_rel not in true_rels:
                                    correct = matched = False
                                    reason.append(
                                        f"expected {classname} {expected_rel} target, found "
                                        + f"{' and '.join(true_rels)} target"
                                    )
                                    break
                            if not matched:
                                break
            if matched:
                matched_groups.append(found_objects)
            else:
                matched_groups.append(None)
        # Check for non-expected objects
        for req in metadata.get("exclude", []):
            classname = req["class"]
            if len(objects.get(classname, [])) >= req["count"]:
                correct = False
                reason.append(
                    f"expected {classname}<{req['count']}, found {len(objects[classname])}"
                )
        return correct, "\n".join(reason)

    def evaluate_reward(image, objects, metadata):
        """
        Evaluate given image using detected objects on the global metadata specifications.
        Assumptions:
        * Metadata combines 'include' clauses with AND, and 'exclude' clauses with OR
        * All clauses are independent, i.e., duplicating a clause has no effect on the correctness
        * CHANGED: Color and position will only be evaluated on the most confidently predicted objects;
            therefore, objects are expected to appear in sorted order
        """
        correct = True
        reason = []
        rewards = []
        matched_groups = []
        # Check for expected objects
        for req in metadata.get("include", []):
            classname = req["class"]
            matched = True
            found_objects = objects.get(classname, [])
            rewards.append(1 - abs(req["count"] - len(found_objects)) / req["count"])
            if len(found_objects) != req["count"]:
                correct = matched = False
                reason.append(
                    f"expected {classname}=={req['count']}, found {len(found_objects)}"
                )
                if "color" in req or "position" in req:
                    rewards.append(0.0)
            else:
                if "color" in req:
                    # Color check
                    colors = color_classification(image, found_objects, classname)
                    rewards.append(
                        1
                        - abs(req["count"] - colors.count(req["color"])) / req["count"]
                    )
                    if colors.count(req["color"]) != req["count"]:
                        correct = matched = False
                        reason.append(
                            f"expected {req['color']} {classname}>={req['count']}, found "
                            + f"{colors.count(req['color'])} {req['color']}; and "
                            + ", ".join(
                                f"{colors.count(c)} {c}" for c in COLORS if c in colors
                            )
                        )
                if "position" in req and matched:
                    # Relative position check
                    expected_rel, target_group = req["position"]
                    if matched_groups[target_group] is None:
                        correct = matched = False
                        reason.append(f"no target for {classname} to be {expected_rel}")
                        rewards.append(0.0)
                    else:
                        for obj in found_objects:
                            for target_obj in matched_groups[target_group]:
                                true_rels = relative_position(obj, target_obj)
                                if expected_rel not in true_rels:
                                    correct = matched = False
                                    reason.append(
                                        f"expected {classname} {expected_rel} target, found "
                                        + f"{' and '.join(true_rels)} target"
                                    )
                                    rewards.append(0.0)
                                    break
                            if not matched:
                                break
                        rewards.append(1.0)
            if matched:
                matched_groups.append(found_objects)
            else:
                matched_groups.append(None)
        reward = sum(rewards) / len(rewards) if rewards else 0
        return correct, reward, "\n".join(reason)

    def evaluate_image(image_pils, metadatas, only_strict):
        results = inference_detector(
            object_detector, [np.array(image_pil) for image_pil in image_pils]
        )
        ret = []
        for result, image_pil, metadata in zip(
            results, image_pils, metadatas, strict=False
        ):
            bbox = result[0] if isinstance(result, tuple) else result
            segm = result[1] if isinstance(result, tuple) and len(result) > 1 else None
            image = ImageOps.exif_transpose(image_pil)
            detected = {}
            # Determine bounding boxes to keep
            confidence_threshold = (
                THRESHOLD if metadata["tag"] != "counting" else COUNTING_THRESHOLD
            )
            for index, classname in enumerate(classnames):
                ordering = np.argsort(bbox[index][:, 4])[::-1]
                ordering = ordering[
                    bbox[index][ordering, 4] > confidence_threshold
                ]  # Threshold
                ordering = ordering[
                    :MAX_OBJECTS
                ].tolist()  # Limit number of detected objects per class
                detected[classname] = []
                while ordering:
                    max_obj = ordering.pop(0)
                    detected[classname].append(
                        (
                            bbox[index][max_obj],
                            None if segm is None else segm[index][max_obj],
                        )
                    )
                    ordering = [
                        obj
                        for obj in ordering
                        if NMS_THRESHOLD == 1
                        or compute_iou(bbox[index][max_obj], bbox[index][obj])
                        < NMS_THRESHOLD
                    ]
                if not detected[classname]:
                    del detected[classname]
            # Evaluate
            is_strict_correct, score, reason = evaluate_reward(
                image, detected, metadata
            )
            if only_strict:
                is_correct = False
            else:
                is_correct, _ = evaluate(image, detected, metadata)
            ret.append(
                {
                    "tag": metadata["tag"],
                    "prompt": metadata["prompt"],
                    "correct": is_correct,
                    "strict_correct": is_strict_correct,
                    "score": score,
                    "reason": reason,
                    "metadata": json.dumps(metadata),
                    "details": json.dumps(
                        {
                            key: [box.tolist() for box, _ in value]
                            for key, value in detected.items()
                        }
                    ),
                }
            )
        return ret

    object_detector, (clip_model, transform, tokenizer), classnames = load_models()
    THRESHOLD = 0.3
    COUNTING_THRESHOLD = 0.9
    MAX_OBJECTS = 16
    NMS_THRESHOLD = 1.0
    POSITION_THRESHOLD = 0.1

    @torch.no_grad()
    def run_geneval(images, metadatas, only_strict=False):
        return evaluate_image(images, metadatas, only_strict=only_strict)

    return run_geneval
