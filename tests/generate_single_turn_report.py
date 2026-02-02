from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import gym_v
from gym_v.envs.registration import registry as ENV_REGISTRY
from tests.test_single_turn_envs import ALL_ENVS


def _get_env_class_name(entry_point: str | None) -> str | None:
    if not entry_point or ":" not in entry_point:
        return None
    return entry_point.split(":", maxsplit=1)[1]


def _get_controller(namespace: str, env_class_name: str, difficulty: int) -> Any | None:
    module_map = {
        "RLVE": "gym_v.envs.rlve.parameter_controllers",
        "GameRL": "gym_v.envs.gamerl.parameter_controllers",
        "ReasoningGym": "gym_v.envs.reasongym.parameter_controllers",
        "Perception": "gym_v.envs.perception.parameter_controllers",
        "Sphinx": "gym_v.envs.sphinx.parameter_controllers",
        "VGRP": "gym_v.envs.vgrp.parameter_controllers",
        "TextArena": "gym_v.envs.textarena.parameter_controllers",
    }
    module_path = module_map.get(namespace)
    if module_path is None:
        return None
    module = __import__(module_path, fromlist=["get_controller_for_env"])
    get_controller = getattr(module, "get_controller_for_env", None)
    if get_controller is None:
        return None
    return get_controller(env_class_name, difficulty)


def _format_params(params: dict[str, Any]) -> str:
    return json.dumps(params, ensure_ascii=False, sort_keys=True)


def main() -> None:
    root = Path(__file__).resolve().parent
    report_root = root
    index_path = report_root / "single_turn_report.html"

    rows: list[str] = []
    for env_id, env_name in ALL_ENVS.items():
        if env_id not in ENV_REGISTRY:
            continue
        suite = env_id.split("/")[0]
        spec = gym_v.spec(env_id)
        env_class_name = _get_env_class_name(
            spec.entry_point if isinstance(spec.entry_point, str) else None
        )
        if env_class_name is None:
            continue

        controller = _get_controller(suite, env_class_name, 0)
        params_d0 = controller.get_parameters() if controller else {}
        controller = _get_controller(suite, env_class_name, 5)
        params_d5 = controller.get_parameters() if controller else {}

        report_dir = None
        output_dir = None
        name = env_id.split("/")[1].replace("-v0", "")
        snake_name = "".join(
            f"_{c.lower()}" if c.isupper() else c for c in name
        ).lstrip("_")
        output_dir = report_root / f"test_output_{suite.lower()}_{snake_name}"
        report_dir = output_dir / "report.html"
        report_link = report_dir.as_posix() if report_dir.exists() else ""

        link_html = (
            f'<a href="{html.escape(report_link)}">report</a>'
            if report_link
            else "missing"
        )
        rows.append(
            "<tr>"
            f"<td>{html.escape(env_id)}</td>"
            f"<td>{html.escape(env_name)}</td>"
            f"<td>{html.escape(suite)}</td>"
            f"<td>{html.escape(_format_params(params_d0))}</td>"
            f"<td>{html.escape(_format_params(params_d5))}</td>"
            f"<td>{link_html}</td>"
            "</tr>"
        )

    content = f"""
    <html>
      <head>
        <meta charset="utf-8" />
        <title>Single-turn Difficulty Report</title>
        <style>
          body {{ font-family: Arial, sans-serif; margin: 24px; }}
          table {{ border-collapse: collapse; width: 100%; }}
          th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
          th {{ background: #f6f6f6; }}
          code {{ background: #f3f3f3; padding: 2px 4px; }}
        </style>
      </head>
      <body>
        <h1>Single-turn 统一报告</h1>
        <p>difficulty 计算方式说明：</p>
        <ul>
          <li>difficulty=0/5 的参数取自对应 suite 的 controller。</li>
          <li>若环境未实现动态难度，则通过 <code>gym_v.make()</code> 的 make-time 参数映射注入。</li>
          <li>表格中给出 <code>difficulty=0/5</code> 的参数快照，便于对比。</li>
        </ul>
        <table>
          <thead>
            <tr>
              <th>Env ID</th>
              <th>Env Name</th>
              <th>Suite</th>
              <th>difficulty=0 参数</th>
              <th>difficulty=5 参数</th>
              <th>HTML 报告</th>
            </tr>
          </thead>
          <tbody>
            {''.join(rows)}
          </tbody>
        </table>
      </body>
    </html>
    """
    index_path.write_text(content, encoding="utf-8")
    print(f"Wrote {index_path}")


if __name__ == "__main__":
    main()
