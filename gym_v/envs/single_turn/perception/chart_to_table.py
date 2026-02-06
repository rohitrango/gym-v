"""Chart to Table perception environment."""

from __future__ import annotations

import io
import logging
import random
import string
from textwrap import dedent
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter

from gym_v import Env, Observation, get_logger

logger = get_logger()

# Suppress matplotlib font warnings
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


class ChartToTableEnv(Env):
    # Meta: source=Perception, category=perception, turn=single
    """Chart extraction environment.

    The agent must perceive a generated chart (bar, line, scatter, pie)
    and extract the underlying data into a structured format (JSON).
    """

    def __init__(
        self,
        img_size: tuple[int, int] = (640, 480),
        max_categories: int = 12,  # Increased for more complex tables
        num_players: int = 1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.img_size = img_size
        self.max_categories = max_categories
        self.num_players = num_players
        self._agent_ids = {f"agent_{i}" for i in range(num_players)}

        self._seed: int | None = None
        self._current_data: dict[str, Any] | None = None
        self._current_chart_type: str | None = None
        self._current_style: str | None = None
        self._current_image: Image.Image | None = None

        # Resources for randomization
        self._hatches = ["/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"]
        self._markers = [
            "o",
            "v",
            "^",
            "<",
            ">",
            "s",
            "p",
            "*",
            "h",
            "H",
            "D",
            "d",
            "P",
            "X",
        ]
        self._linestyles = ["-", "--", "-.", ":"]

        # Vivid colors for high visibility
        self._vivid_colors = [
            "#FF0000",
            "#00FF00",
            "#0000FF",
            "#FFFF00",
            "#FF00FF",
            "#00FFFF",
            "#FF4500",
            "#32CD32",
            "#1E90FF",
            "#FFD700",
            "#FF1493",
            "#00CED1",
            "#FF6347",
            "#7CFC00",
            "#4169E1",
            "#FFA500",
            "#C71585",
            "#40E0D0",
        ]

    @property
    def description(self) -> str:
        return dedent("""
            You are given a chart image.
            Your task is to extract the data presented in the chart and output it as a JSON object.

            The JSON should be a dictionary where keys are the category names (labels) and values are the numerical values.
            If there are multiple series, the values will be dictionaries or lists.

            Example (Single Series): {"A": 10, "B": 24}
            Example (Multi Series): {"2020": {"Revenue": 100, "Cost": 80}, "2021": {"Revenue": 120, "Cost": 90}}

            Output ONLY the JSON string.
        """).strip()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, Observation], dict[str, Any]]:
        super().reset(seed=seed)
        self._seed = seed
        # Seeding numpy and random
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self._generate_new_problem()

        obs = Observation(
            image=self._current_image,
            text=None,
            metadata={
                "chart_type": self._current_chart_type,
                "style": self._current_style,
            },
        )

        info = {
            "oracle_answer": str(self._current_data),
            "chart_type": self._current_chart_type,
            "style": self._current_style,
        }

        return {agent_id: obs for agent_id in self._agent_ids}, {
            agent_id: info for agent_id in self._agent_ids
        }

    def inner_step(
        self, action: dict[str, str]
    ) -> tuple[
        dict[str, Observation],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, Any],
    ]:
        agent_id = next(iter(self._agent_ids))
        action_str = action[agent_id]
        reward = self._compute_reward(action_str)

        info = {"oracle_answer": str(self._current_data)}

        obs = Observation(image=self._current_image, text=None)

        return (
            {agent_id: obs for agent_id in self._agent_ids},
            {agent_id: reward for agent_id in self._agent_ids},
            {
                **{agent_id: True for agent_id in self._agent_ids},
                "__all__": True,
            },
            {
                **{agent_id: False for agent_id in self._agent_ids},
                "__all__": False,
            },
            {agent_id: info for agent_id in self._agent_ids},
        )

    def _compute_reward(self, action: str) -> float:
        """Compute reward by comparing action with oracle answer."""
        try:
            # Try to parse the action as Python dict or JSON
            try:
                parsed_action = eval(action)
            except Exception:
                import json

                parsed_action = json.loads(action)

            # Compare with oracle data
            if parsed_action == self._current_data:
                return 1.0
            return 0.0
        except Exception:
            return 0.0

    def render(self) -> Image.Image:
        return self._current_image

    def _generate_new_problem(self):
        """Generates random data and renders it."""

        # Decide if single series or multi-series (stacked/grouped)
        is_multi_series = random.random() < 0.4

        num_items = random.randint(3, self.max_categories)

        # Generate random labels
        if random.random() < 0.5:
            labels = [f"Item {string.ascii_uppercase[i]}" for i in range(num_items)]
        else:
            # Maybe simple dates or short words
            years = list(range(2015, 2030))
            start_year = random.choice(years)
            labels = [str(start_year + i) for i in range(num_items)]

        if is_multi_series:
            series_names = (
                ["Series A", "Series B"]
                if random.random() < 0.7
                else ["Revenue", "Cost", "Profit"][: random.randint(2, 3)]
            )
            data = {}
            # Organize data by label -> {series: value}
            raw_values = []  # For plotting
            for label in labels:
                label_data = {}
                series_vals = []
                for s_name in series_names:
                    val = int(np.random.randint(10, 100))
                    label_data[s_name] = val
                    series_vals.append(val)
                data[label] = label_data
                raw_values.append(series_vals)

            self._current_data = data
            values = np.array(raw_values).T  # Shape: (num_series, num_items)
            self._current_series_names = series_names
        else:
            # Single series
            values = np.random.randint(10, 100, size=num_items).tolist()
            self._current_data = dict(zip(labels, values, strict=False))
            self._current_series_names = None

        # Pick a chart type
        if is_multi_series:
            chart_types = ["bar_grouped", "bar_stacked", "line_multi", "area_stacked"]
        else:
            chart_types = ["bar", "line", "scatter", "pie", "area", "donut"]

        self._current_chart_type = random.choice(chart_types)

        # Pick a style
        styles = [
            "default",
            "ggplot",
            "bmh",
            "seaborn-v0_8-dark",
            "xkcd",
            "grayscale",
            "classic",
            "dark_background",
            "Solarize_Light2",
        ]
        # Filter available styles in matplotlib
        available_styles = plt.style.available
        candidates = [s for s in styles if s in available_styles] + ["default", "xkcd"]
        self._current_style = random.choice(candidates)

        try:
            self._current_image = self._render_chart(
                labels, values, self._current_chart_type, self._current_style
            )
        except Exception as e:
            logger.warning(
                f"Render failed with style {self._current_style}: {e}. Retrying with default."
            )
            self._current_style = "default"
            self._current_image = self._render_chart(
                labels, values, self._current_chart_type, "default"
            )

    def _render_chart(self, labels, values, chart_type, style) -> Image.Image:
        # Context manager for style
        img = None
        try:
            if style == "xkcd":
                with plt.xkcd():
                    img = self._plot(labels, values, chart_type, is_xkcd=True)
            elif style == "default":
                with plt.style.context("default"):
                    img = self._plot(labels, values, chart_type)
            else:
                with plt.style.context(style):
                    img = self._plot(labels, values, chart_type)
        except Exception:
            # If style fails, try default
            with plt.style.context("default"):
                img = self._plot(labels, values, chart_type)

        # Post-processing: Add random background noise or tint
        if img and random.random() < 0.3:
            img = self._add_noise(img)

        return img

    def _get_colors(self, num_colors, style):
        """Returns a list of colors based on style preference."""
        if style == "grayscale":
            return [plt.cm.gray(i) for i in np.linspace(0.2, 0.8, num_colors)]

        # 50% chance to use vivid colors, otherwise use colormap
        if random.random() < 0.5:
            # Sample from vivid colors
            return random.sample(
                self._vivid_colors * (num_colors // len(self._vivid_colors) + 1),
                num_colors,
            )
        else:
            cmap_name = random.choice(
                [
                    "viridis",
                    "plasma",
                    "inferno",
                    "magma",
                    "cividis",
                    "tab10",
                    "Set3",
                    "hsv",
                    "nipy_spectral",
                ]
            )
            cmap = plt.get_cmap(cmap_name)
            colors = [cmap(i) for i in np.linspace(0, 1, num_colors)]
            random.shuffle(colors)
            return colors

    def _plot(self, labels, values, chart_type, is_xkcd=False) -> Image.Image:
        fig = plt.figure(
            figsize=(self.img_size[0] / 100, self.img_size[1] / 100), dpi=100
        )
        ax = fig.add_subplot(111)

        # Randomize grid
        if random.random() < 0.7:
            style = "-" if is_xkcd else random.choice(["-", "--", "-.", ":"])
            ax.grid(True, linestyle=style, alpha=random.uniform(0.3, 0.7))
        else:
            ax.grid(False)

        # Handle multi-series logic
        if chart_type in ["bar_grouped", "bar_stacked", "line_multi", "area_stacked"]:
            num_series = values.shape[0]
            num_items = values.shape[1]
            colors = self._get_colors(num_series, self._current_style)
            series_names = self._current_series_names

            if chart_type == "bar_grouped":
                width = 0.8 / num_series
                x = np.arange(len(labels))
                for i in range(num_series):
                    offset = (i - num_series / 2 + 0.5) * width
                    hatch = (
                        random.choice(self._hatches)
                        if random.random() < 0.3 and not is_xkcd
                        else None
                    )
                    bars = ax.bar(
                        x + offset,
                        values[i],
                        width,
                        label=series_names[i],
                        color=colors[i],
                        edgecolor="black",
                    )
                    if hatch:
                        for bar in bars:
                            bar.set_hatch(hatch)
                ax.set_xticks(x)
                ax.set_xticklabels(labels)

            elif chart_type == "bar_stacked":
                bottom = np.zeros(num_items)
                for i in range(num_series):
                    hatch = (
                        random.choice(self._hatches)
                        if random.random() < 0.3 and not is_xkcd
                        else None
                    )
                    bars = ax.bar(
                        labels,
                        values[i],
                        bottom=bottom,
                        label=series_names[i],
                        color=colors[i],
                        edgecolor="black",
                    )
                    if hatch:
                        for bar in bars:
                            bar.set_hatch(hatch)
                    bottom += values[i]

            elif chart_type == "line_multi":
                for i in range(num_series):
                    marker = random.choice(self._markers)
                    linestyle = "-" if is_xkcd else random.choice(self._linestyles)
                    ax.plot(
                        labels,
                        values[i],
                        label=series_names[i],
                        color=colors[i],
                        marker=marker,
                        linestyle=linestyle,
                        linewidth=2,
                    )

            elif chart_type == "area_stacked":
                ax.stackplot(
                    labels, values, labels=series_names, colors=colors, alpha=0.7
                )

            # Add legend
            ax.legend(loc="best")

        else:
            # Single series logic
            colors = self._get_colors(len(values), self._current_style)

            if chart_type == "bar":
                hatch = (
                    random.choice(self._hatches)
                    if random.random() < 0.3 and not is_xkcd
                    else None
                )
                if random.random() < 0.3:
                    bars = ax.barh(
                        labels,
                        values,
                        color=colors,
                        edgecolor="black" if hatch else None,
                    )
                else:
                    bars = ax.bar(
                        labels,
                        values,
                        color=colors,
                        edgecolor="black" if hatch else None,
                    )
                if hatch:
                    for bar in bars:
                        bar.set_hatch(hatch)

            elif chart_type == "line":
                marker = random.choice(self._markers)
                linestyle = "-" if is_xkcd else random.choice(self._linestyles)
                color = colors[0]
                if random.random() < 0.3:
                    ax.fill_between(labels, values, alpha=0.3, color=color)
                ax.plot(
                    labels,
                    values,
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=random.randint(2, 5),
                    color=color,
                )

            elif chart_type == "scatter":
                marker = random.choice(self._markers)
                sizes = np.random.randint(80, 300, size=len(values))
                alpha = 0.6 if random.random() < 0.5 else 1.0
                ax.scatter(
                    labels,
                    values,
                    s=sizes,
                    c=colors,
                    marker=marker,
                    alpha=alpha,
                    edgecolors="black",
                )

            elif chart_type == "pie" or chart_type == "donut":
                explode = [0] * len(values)
                if random.random() < 0.3:
                    explode[random.randint(0, len(values) - 1)] = 0.1
                wedgeprops = dict(width=0.4) if chart_type == "donut" else None
                wedges, texts, autotexts = ax.pie(
                    values,
                    labels=labels,
                    autopct="%1.1f%%",
                    explode=explode,
                    colors=colors,
                    shadow=(random.random() < 0.5 and not is_xkcd),
                    wedgeprops=wedgeprops,
                    pctdistance=0.85 if chart_type == "donut" else 0.6,
                )
                if chart_type == "donut":
                    fig.gca().add_artist(plt.Circle((0, 0), 0.70, fc="white"))
                if random.random() < 0.4 and not is_xkcd:
                    for w in wedges:
                        w.set_hatch(random.choice(self._hatches))
                        w.set_edgecolor("black")

            elif chart_type == "area":
                ax.stackplot(labels, values, colors=colors, alpha=0.6)
                ax.plot(labels, values, color="black", alpha=0.3)

        # Randomize titles and labels
        if random.random() < 0.9:
            titles = [
                "Sales Report",
                "Growth Analysis",
                "Distribution",
                "Summary",
                "Data Overview",
                "Monthly Metrics",
                "Key Performance",
            ]
            ax.set_title(
                random.choice(titles),
                fontsize=random.randint(14, 20),
                fontweight="bold",
            )

        if chart_type not in ["pie", "donut"]:
            if random.random() < 0.8:
                ax.set_ylabel("Value (Units)", fontsize=12)
            if random.random() < 0.8:
                ax.set_xlabel("Category", fontsize=12)

            rotation = random.choice([0, 45, 90])
            plt.xticks(rotation=rotation)

            # Value labels for single series
            if chart_type not in [
                "bar_grouped",
                "bar_stacked",
                "line_multi",
                "area_stacked",
            ]:
                if random.random() < 0.3 and chart_type in ["bar", "line", "scatter"]:
                    for i, v in enumerate(values):
                        ax.text(
                            i,
                            v + 1,
                            str(v),
                            ha="center",
                            va="bottom",
                            fontweight="bold",
                        )

        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        buf.seek(0)
        img = Image.open(buf).convert("RGB")
        plt.close(fig)

        return img

    def _add_noise(self, img: Image.Image) -> Image.Image:
        """Adds salt-and-pepper noise or blur."""
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
        if random.random() < 0.5:
            arr = np.array(img)
            noise = np.random.randint(0, 255, arr.shape, dtype="uint8")
            mask = np.random.rand(*arr.shape[:2]) < 0.05
            arr[mask] = noise[mask]
            img = Image.fromarray(arr)
        return img
