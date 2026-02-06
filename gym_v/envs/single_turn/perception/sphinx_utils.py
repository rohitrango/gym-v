"""Utility functions for Sphinx environment image processing."""

from __future__ import annotations

from importlib import resources
import re
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 8 standard geometric transformations
TRANSFORMS = [
    "identity",
    "rot90_cw",
    "rot180",
    "rot90_ccw",
    "flip_h",
    "flip_v",
    "flip_diag",
    "flip_antidiag",
]

# Available visual styles for polygon generation
# Ordered by complexity (difficulty)
POLY_STYLES = [
    "outline",  # Level 1: Simple outline (easiest)
    "filled",  # Level 2: Solid color fill
    "nested",  # Level 3: Multiple nested shapes
    "striped",  # Level 4: Striped texture
    "gradient",  # Level 5: Gradient fill
    "3d",  # Level 6: 3D shadow effect
    "composite",  # Level 7: Multiple shapes combined
    "pixelated",  # Level 8: Pixelated/mosaic style (hardest)
]

# Available visual styles for icon generation
ICON_STYLES = [
    "simple",  # Level 1: Simple geometric icons
    "colored",  # Level 2: Colored icons
    "nested",  # Level 3: Nested shapes
    "complex",  # Level 4: Complex icons (gears, etc.)
]

# Color palette for grid generation (ARC/Sphinx style)
GRID_COLORS = [
    (0, 0, 0),  # black
    (0, 116, 217),  # blue
    (255, 65, 54),  # red
    (46, 204, 64),  # green
    (255, 220, 0),  # yellow
    (170, 170, 170),  # gray
    (240, 18, 190),  # pink/magenta
    (255, 133, 27),  # orange
    (127, 219, 255),  # cyan
    (135, 12, 37),  # maroon
]

# Mapping from problem text patterns to transform names
TRANSFORM_PATTERNS = {
    r"rotate 90[°]?\s*clockwise": "rot90_cw",
    r"rotate 270[°]?\s*(?:counter)?clockwise": "rot90_ccw",
    r"rotate 90[°]?\s*counter\s*clockwise": "rot90_ccw",
    r"rotate 90[°]?\s*CCW": "rot90_ccw",
    r"rotate 180[°]?": "rot180",
    r"reflect across a horizontal line": "flip_h",
    r"horizontal line symmetry": "flip_h",
    r"reflect across a vertical line": "flip_v",
    r"vertical line symmetry": "flip_v",
    r"reflect across the main diagonal": "flip_diag",
    r"main.diagonal mirror": "flip_diag",
    r"reflect across the anti.diagonal": "flip_antidiag",
    r"anti.diagonal mirror": "flip_antidiag",
}


def parse_transform_from_problem(problem_text: str) -> str | None:
    """Parse the transformation type from problem text.

    Args:
        problem_text: The problem description text

    Returns:
        Transform name (e.g., 'rot90_cw') or None if not recognized
    """
    problem_lower = problem_text.lower()
    for pattern, transform in TRANSFORM_PATTERNS.items():
        if re.search(pattern, problem_lower, re.IGNORECASE):
            return transform
    return None


def apply_transform(img: Image.Image, transform: str) -> Image.Image:
    """Apply a geometric transformation to an image.

    Args:
        img: Input PIL Image
        transform: One of TRANSFORMS

    Returns:
        Transformed PIL Image
    """
    if transform == "identity":
        return img.copy()
    elif transform == "rot90_cw":
        return img.transpose(Image.Transpose.ROTATE_270)
    elif transform == "rot180":
        return img.transpose(Image.Transpose.ROTATE_180)
    elif transform == "rot90_ccw":
        return img.transpose(Image.Transpose.ROTATE_90)
    elif transform == "flip_h":
        return img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    elif transform == "flip_v":
        return img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    elif transform == "flip_diag":
        return img.transpose(Image.Transpose.TRANSPOSE)
    elif transform == "flip_antidiag":
        return img.transpose(Image.Transpose.TRANSVERSE)
    else:
        raise ValueError(f"Unknown transform: {transform}")


def generate_random_grid(
    rng: np.random.Generator,
    grid_size: int = 5,
    num_colors: int = 4,
    cell_size: int = 40,
    border_width: int = 1,
) -> Image.Image:
    """Generate a random colored grid pattern (ARC/Sphinx style).

    Args:
        rng: numpy random generator for reproducibility
        grid_size: Grid dimensions (grid_size x grid_size)
        num_colors: Number of colors to use from the palette
        cell_size: Pixel size of each cell
        border_width: Width of grid lines

    Returns:
        PIL Image of the colored grid
    """
    # Select colors from palette (skip black at index 0 for background contrast)
    available_colors = GRID_COLORS[1:]  # Exclude black
    if num_colors > len(available_colors):
        num_colors = len(available_colors)
    color_indices = rng.choice(len(available_colors), size=num_colors, replace=False)
    selected_colors = [available_colors[i] for i in color_indices]

    # Generate random color grid
    grid = rng.integers(0, num_colors, size=(grid_size, grid_size))

    # Calculate image dimensions
    img_size = grid_size * cell_size + (grid_size + 1) * border_width

    # Create image with black background (for grid lines)
    img = Image.new("RGB", (img_size, img_size), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Draw cells
    for row in range(grid_size):
        for col in range(grid_size):
            x1 = border_width + col * (cell_size + border_width)
            y1 = border_width + row * (cell_size + border_width)
            x2 = x1 + cell_size
            y2 = y1 + cell_size
            color = selected_colors[grid[row, col]]
            draw.rectangle([x1, y1, x2 - 1, y2 - 1], fill=color)

    return img


def generate_symmetric_2x2_grid(
    rng: np.random.Generator,
    cell_grid_size: int = 4,
    num_colors: int = 3,
    cell_size: int = 100,
) -> tuple[list[Image.Image], int]:
    """Generate a 2x2 grid satisfying vertical + horizontal mirror symmetry.

    For a 2x2 grid with V+H symmetry:
    - cell[0,0] (top-left) and cell[0,1] (top-right) are horizontal mirrors
    - cell[0,0] (top-left) and cell[1,0] (bottom-left) are vertical mirrors
    - cell[1,1] (bottom-right) = rot180(cell[0,0])

    Args:
        rng: numpy random generator for reproducibility
        cell_grid_size: Grid size within each cell
        num_colors: Number of colors to use
        cell_size: Pixel size of each cell

    Returns:
        Tuple of (list of 4 cell images in order [TL, TR, BL, BR], hidden cell index)
    """
    border_width = 1

    # Generate the top-left cell
    top_left = generate_random_grid(
        rng,
        grid_size=cell_grid_size,
        num_colors=num_colors,
        cell_size=cell_size // cell_grid_size,
        border_width=border_width,
    )

    # Generate other cells through symmetry transformations
    top_right = apply_transform(top_left, "flip_h")  # Horizontal mirror
    bottom_left = apply_transform(top_left, "flip_v")  # Vertical mirror
    bottom_right = apply_transform(top_left, "rot180")  # Both mirrors = 180° rotation

    cells = [top_left, top_right, bottom_left, bottom_right]

    # Randomly select which cell to hide
    hidden_idx = int(rng.integers(0, 4))

    return cells, hidden_idx


def _generate_polygon_points(
    rng: np.random.Generator,
    img_size: int,
    num_points: int = 8,
    margin_ratio: float = 0.15,
) -> list[tuple[float, float]]:
    """Generate random polygon vertices using polar coordinates.

    Args:
        rng: numpy random generator
        img_size: Size of the image
        num_points: Number of vertices
        margin_ratio: Margin from edge as ratio of img_size

    Returns:
        List of (x, y) tuples for polygon vertices
    """
    margin = img_size * margin_ratio
    center = img_size / 2
    max_radius = (img_size / 2) - margin

    angles = np.sort(rng.uniform(0, 2 * np.pi, num_points))
    radii = rng.uniform(max_radius * 0.4, max_radius, num_points)

    points = []
    for angle, radius in zip(angles, radii, strict=True):
        x = center + radius * np.cos(angle)
        y = center + radius * np.sin(angle)
        points.append((x, y))

    return points


def _draw_grid_background(
    draw: ImageDraw.ImageDraw,
    img_size: int,
    divisions: int = 8,
    color: tuple[int, int, int] = (200, 200, 200),
) -> None:
    """Draw grid lines on the background."""
    cell_size = img_size // divisions
    for i in range(divisions + 1):
        pos = i * cell_size
        draw.line([(pos, 0), (pos, img_size)], fill=color, width=1)
        draw.line([(0, pos), (img_size, pos)], fill=color, width=1)


def _style_outline(
    rng: np.random.Generator,
    img_size: int,
    num_points: int,
    line_width: int,
    grid_lines: bool,
    grid_divisions: int,
) -> Image.Image:
    """Style 1: Simple outline polygon."""
    img = Image.new("RGB", (img_size, img_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    if grid_lines:
        _draw_grid_background(draw, img_size, grid_divisions)

    points = _generate_polygon_points(rng, img_size, num_points)
    shape_color = (80, 80, 80)
    draw.polygon(points, outline=shape_color, width=line_width)
    for i in range(len(points)):
        p1 = points[i]
        p2 = points[(i + 1) % len(points)]
        draw.line([p1, p2], fill=shape_color, width=line_width)

    return img


def _style_filled(
    rng: np.random.Generator,
    img_size: int,
    num_points: int,
    line_width: int,
    grid_lines: bool,
    grid_divisions: int,
) -> Image.Image:
    """Style 2: Solid color filled polygon."""
    img = Image.new("RGB", (img_size, img_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    if grid_lines:
        _draw_grid_background(draw, img_size, grid_divisions)

    points = _generate_polygon_points(rng, img_size, num_points)

    # Random fill color from palette
    fill_colors = GRID_COLORS[1:7]  # Skip black, use vibrant colors
    fill_color = fill_colors[int(rng.integers(0, len(fill_colors)))]
    outline_color = tuple(max(0, c - 50) for c in fill_color)

    draw.polygon(points, fill=fill_color, outline=outline_color, width=line_width)

    return img


def _style_nested(
    rng: np.random.Generator,
    img_size: int,
    num_points: int,
    line_width: int,
    grid_lines: bool,
    grid_divisions: int,
) -> Image.Image:
    """Style 3: Multiple nested shapes."""
    img = Image.new("RGB", (img_size, img_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    if grid_lines:
        _draw_grid_background(draw, img_size, grid_divisions)

    # Use same seed for consistent shape across layers
    base_points = _generate_polygon_points(rng, img_size, num_points)
    center = img_size / 2

    colors = [(255, 65, 54), (255, 220, 0), (46, 204, 64)]
    scales = [1.0, 0.65, 0.35]

    for scale, color in zip(scales, colors, strict=True):
        scaled_pts = [
            (center + (x - center) * scale, center + (y - center) * scale)
            for x, y in base_points
        ]
        draw.polygon(scaled_pts, fill=color, outline=(40, 40, 40), width=line_width)

    return img


def _style_striped(
    rng: np.random.Generator,
    img_size: int,
    num_points: int,
    line_width: int,
    grid_lines: bool,
    grid_divisions: int,
) -> Image.Image:
    """Style 4: Striped texture fill."""
    img = Image.new("RGB", (img_size, img_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    if grid_lines:
        _draw_grid_background(draw, img_size, grid_divisions)

    points = _generate_polygon_points(rng, img_size, num_points)

    # Fill with light color
    fill_color = (220, 240, 255)
    line_color = (0, 100, 180)
    draw.polygon(points, fill=fill_color, outline=line_color, width=line_width)

    # Draw diagonal stripes across entire image
    stripe_spacing = 15
    for i in range(-img_size, img_size * 2, stripe_spacing):
        draw.line([(i, 0), (i + img_size, img_size)], fill=line_color, width=1)

    # Redraw outline to clean edges
    draw.polygon(points, outline=line_color, width=line_width)

    return img


def _style_gradient(
    rng: np.random.Generator,
    img_size: int,
    num_points: int,
    line_width: int,
    grid_lines: bool,
    grid_divisions: int,
) -> Image.Image:
    """Style 5: Gradient fill."""
    img = Image.new("RGB", (img_size, img_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    if grid_lines:
        _draw_grid_background(draw, img_size, grid_divisions)

    points = _generate_polygon_points(rng, img_size, num_points)

    # Create gradient
    gradient = Image.new("RGB", (img_size, img_size))
    for y in range(img_size):
        r = int(255 - (y / img_size) * 100)
        g = int(100 + (y / img_size) * 100)
        b = int(150 + (y / img_size) * 50)
        for x in range(img_size):
            gradient.putpixel((x, y), (r, g, b))

    # Create mask from polygon
    mask = Image.new("L", (img_size, img_size), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.polygon(points, fill=255)

    # Apply gradient through mask
    img.paste(gradient, mask=mask)
    draw = ImageDraw.Draw(img)
    draw.polygon(points, outline=(50, 50, 50), width=line_width)

    return img


def _style_3d(
    rng: np.random.Generator,
    img_size: int,
    num_points: int,
    line_width: int,
    grid_lines: bool,
    grid_divisions: int,
) -> Image.Image:
    """Style 6: 3D shadow effect."""
    img = Image.new("RGB", (img_size, img_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    if grid_lines:
        _draw_grid_background(draw, img_size, grid_divisions)

    points = _generate_polygon_points(rng, img_size, num_points)

    # Draw shadow offset
    offset = 15
    shadow_points = [(x + offset, y + offset) for x, y in points]
    draw.polygon(shadow_points, fill=(100, 100, 100))

    # Draw main shape
    fill_color = (0, 180, 230)
    draw.polygon(points, fill=fill_color, outline=(0, 100, 150), width=line_width)

    return img


def _style_composite(
    rng: np.random.Generator,
    img_size: int,
    num_points: int,
    line_width: int,
    grid_lines: bool,
    grid_divisions: int,
) -> Image.Image:
    """Style 7: Multiple shapes combined."""
    img = Image.new("RGB", (img_size, img_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    if grid_lines:
        _draw_grid_background(draw, img_size, grid_divisions)

    margin = img_size * 0.15
    colors = [(255, 65, 54), (0, 116, 217), (46, 204, 64)]

    # Draw multiple random shapes
    shapes_data = [
        ("ellipse", rng.uniform(margin, img_size * 0.4, 4)),
        ("rectangle", rng.uniform(margin, img_size * 0.6, 4)),
        ("polygon", _generate_polygon_points(rng, img_size, num_points // 2 + 2)),
    ]

    for i, (shape_type, data) in enumerate(shapes_data):
        color = colors[i % len(colors)]
        if shape_type == "ellipse":
            x1, y1, x2, y2 = sorted(data[:2]) + sorted(data[2:])
            x2 = min(x2 + img_size * 0.3, img_size - margin)
            y2 = min(y2 + img_size * 0.2, img_size - margin)
            draw.ellipse([x1, y1, x2, y2], fill=color, outline=(40, 40, 40), width=2)
        elif shape_type == "rectangle":
            x1, y1 = data[0], data[1]
            x2 = min(x1 + img_size * 0.35, img_size - margin)
            y2 = min(y1 + img_size * 0.3, img_size - margin)
            draw.rectangle([x1, y1, x2, y2], fill=color, outline=(40, 40, 40), width=2)
        else:
            # Offset polygon to different position
            offset_x = rng.uniform(-margin, margin)
            offset_y = rng.uniform(-margin, margin)
            pts = [(x + offset_x, y + offset_y) for x, y in data]
            draw.polygon(pts, fill=color, outline=(40, 40, 40), width=2)

    return img


def _style_pixelated(
    rng: np.random.Generator,
    img_size: int,
    num_points: int,
    line_width: int,
    grid_lines: bool,
    grid_divisions: int,
) -> Image.Image:
    """Style 8: Pixelated/mosaic style."""
    img = Image.new("RGB", (img_size, img_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    points = _generate_polygon_points(rng, img_size, num_points)

    # Create mask
    mask = Image.new("L", (img_size, img_size), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.polygon(points, fill=255)

    # Pixelate
    pixel_size = 15
    colors = [(240, 18, 190), (255, 133, 27)]

    for x in range(0, img_size, pixel_size):
        for y in range(0, img_size, pixel_size):
            cx, cy = x + pixel_size // 2, y + pixel_size // 2
            if cx < img_size and cy < img_size and mask.getpixel((cx, cy)) > 128:
                color = colors[(x // pixel_size + y // pixel_size) % 2]
                draw.rectangle(
                    [x + 1, y + 1, x + pixel_size - 1, y + pixel_size - 1],
                    fill=color,
                    outline=(40, 40, 40),
                )

    return img


def generate_random_polygon(
    rng: np.random.Generator,
    img_size: int = 300,
    num_points: int = 8,
    line_width: int = 3,
    grid_lines: bool = True,
    grid_divisions: int = 8,
    style: str | None = None,
    difficulty: int | None = None,
) -> Image.Image:
    """Generate a random closed polygon shape with various visual styles.

    This creates shapes similar to the original Sphinx Transform Result task.
    The style can be specified directly, chosen by difficulty level, or random.

    Args:
        rng: numpy random generator for reproducibility
        img_size: Size of the output image (square)
        num_points: Number of vertices in the polygon
        line_width: Width of the polygon lines
        grid_lines: Whether to draw grid lines in the background
        grid_divisions: Number of grid divisions
        style: Visual style name from POLY_STYLES, or "random" for random selection.
               If None, uses difficulty to select style.
        difficulty: Difficulty level 1-8 (maps to styles in order).
                   Higher = more complex visual style. Only used if style is None.

    Returns:
        PIL Image with the polygon shape
    """
    # Style selection logic
    if style == "random" or (style is None and difficulty is None):
        selected_style = POLY_STYLES[int(rng.integers(0, len(POLY_STYLES)))]
    elif style is not None and style != "random":
        if style not in POLY_STYLES:
            raise ValueError(f"Unknown style: {style}. Choose from {POLY_STYLES}")
        selected_style = style
    else:
        # Use difficulty to select style
        difficulty = max(1, min(difficulty, len(POLY_STYLES)))
        # Allow random selection from styles up to difficulty level
        max_style_idx = difficulty
        selected_style = POLY_STYLES[int(rng.integers(0, max_style_idx))]

    # Dispatch to style-specific function
    style_funcs = {
        "outline": _style_outline,
        "filled": _style_filled,
        "nested": _style_nested,
        "striped": _style_striped,
        "gradient": _style_gradient,
        "3d": _style_3d,
        "composite": _style_composite,
        "pixelated": _style_pixelated,
    }

    return style_funcs[selected_style](
        rng, img_size, num_points, line_width, grid_lines, grid_divisions
    )


def _icon_simple(
    rng: np.random.Generator,
    img_size: int,
    line_width: int,
    icon_color: tuple[int, int, int],
) -> Image.Image:
    """Simple geometric icon (outline only)."""
    img = Image.new("RGB", (img_size, img_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    margin = img_size * 0.1
    inner_size = img_size - 2 * margin

    icon_type = int(rng.integers(0, 5))

    if icon_type == 0:
        # Rectangle with lines
        x1, y1 = margin, margin
        x2, y2 = img_size - margin, img_size - margin
        draw.rounded_rectangle(
            [x1, y1, x2, y2], radius=10, outline=icon_color, width=line_width
        )
        num_lines = int(rng.integers(2, 5))
        for i in range(num_lines):
            ly = y1 + (i + 1) * inner_size / (num_lines + 1)
            lx1 = x1 + inner_size * 0.15
            lx2 = x2 - inner_size * 0.15
            draw.line([(lx1, ly), (lx2, ly)], fill=icon_color, width=line_width - 1)
    elif icon_type == 1:
        # Star
        center = img_size / 2
        outer_r = inner_size / 2
        inner_r = outer_r * 0.4
        num_spikes = int(rng.integers(4, 8))
        points = []
        for i in range(num_spikes * 2):
            angle = i * np.pi / num_spikes - np.pi / 2
            r = outer_r if i % 2 == 0 else inner_r
            x = center + r * np.cos(angle)
            y = center + r * np.sin(angle)
            points.append((x, y))
        draw.polygon(points, outline=icon_color, width=line_width)
    elif icon_type == 2:
        # Nested rectangles
        num_rects = int(rng.integers(2, 4))
        for i in range(num_rects):
            offset = margin + i * inner_size / (num_rects * 2)
            x1, y1 = offset, offset
            x2, y2 = img_size - offset, img_size - offset
            draw.rectangle([x1, y1, x2, y2], outline=icon_color, width=line_width)
    elif icon_type == 3:
        # Arrow
        cx, cy = img_size / 2, img_size / 2
        size = inner_size * 0.4
        direction = int(rng.integers(0, 4))
        if direction == 0:
            points = [
                (cx - size, cy - size / 2),
                (cx, cy - size / 2),
                (cx, cy - size),
                (cx + size, cy),
                (cx, cy + size),
                (cx, cy + size / 2),
                (cx - size, cy + size / 2),
            ]
        elif direction == 1:
            points = [
                (cx - size / 2, cy - size),
                (cx + size / 2, cy - size),
                (cx + size / 2, cy),
                (cx + size, cy),
                (cx, cy + size),
                (cx - size, cy),
                (cx - size / 2, cy),
            ]
        elif direction == 2:
            points = [
                (cx + size, cy - size / 2),
                (cx, cy - size / 2),
                (cx, cy - size),
                (cx - size, cy),
                (cx, cy + size),
                (cx, cy + size / 2),
                (cx + size, cy + size / 2),
            ]
        else:
            points = [
                (cx - size / 2, cy + size),
                (cx + size / 2, cy + size),
                (cx + size / 2, cy),
                (cx + size, cy),
                (cx, cy - size),
                (cx - size, cy),
                (cx - size / 2, cy),
            ]
        draw.polygon(points, outline=icon_color, width=line_width)
    else:
        # Circle with cross
        x1, y1 = margin, margin
        x2, y2 = img_size - margin, img_size - margin
        draw.ellipse([x1, y1, x2, y2], outline=icon_color, width=line_width)
        cx, cy = img_size / 2, img_size / 2
        r = inner_size / 2 * 0.6
        draw.line([(cx - r, cy), (cx + r, cy)], fill=icon_color, width=line_width)
        draw.line([(cx, cy - r), (cx, cy + r)], fill=icon_color, width=line_width)

    return img


def _icon_colored(
    rng: np.random.Generator,
    img_size: int,
    line_width: int,
    icon_color: tuple[int, int, int],
) -> Image.Image:
    """Colored/filled icon."""
    img = Image.new("RGB", (img_size, img_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    margin = img_size * 0.1
    inner_size = img_size - 2 * margin

    # Lighter fill color
    fill_color = tuple(min(255, c + 100) for c in icon_color)
    outline_color = icon_color

    icon_type = int(rng.integers(0, 4))

    if icon_type == 0:
        # Filled rectangle
        x1, y1 = margin, margin
        x2, y2 = img_size - margin, img_size - margin
        draw.rounded_rectangle(
            [x1, y1, x2, y2],
            radius=15,
            fill=fill_color,
            outline=outline_color,
            width=line_width,
        )
    elif icon_type == 1:
        # Filled star
        center = img_size / 2
        outer_r = inner_size / 2
        inner_r = outer_r * 0.4
        num_spikes = int(rng.integers(4, 8))
        points = []
        for i in range(num_spikes * 2):
            angle = i * np.pi / num_spikes - np.pi / 2
            r = outer_r if i % 2 == 0 else inner_r
            x = center + r * np.cos(angle)
            y = center + r * np.sin(angle)
            points.append((x, y))
        draw.polygon(points, fill=fill_color, outline=outline_color, width=line_width)
    elif icon_type == 2:
        # Filled arrow
        cx, cy = img_size / 2, img_size / 2
        size = inner_size * 0.4
        direction = int(rng.integers(0, 4))
        if direction == 0:
            points = [
                (cx - size, cy - size / 2),
                (cx, cy - size / 2),
                (cx, cy - size),
                (cx + size, cy),
                (cx, cy + size),
                (cx, cy + size / 2),
                (cx - size, cy + size / 2),
            ]
        elif direction == 1:
            points = [
                (cx - size / 2, cy - size),
                (cx + size / 2, cy - size),
                (cx + size / 2, cy),
                (cx + size, cy),
                (cx, cy + size),
                (cx - size, cy),
                (cx - size / 2, cy),
            ]
        elif direction == 2:
            points = [
                (cx + size, cy - size / 2),
                (cx, cy - size / 2),
                (cx, cy - size),
                (cx - size, cy),
                (cx, cy + size),
                (cx, cy + size / 2),
                (cx + size, cy + size / 2),
            ]
        else:
            points = [
                (cx - size / 2, cy + size),
                (cx + size / 2, cy + size),
                (cx + size / 2, cy),
                (cx + size, cy),
                (cx, cy - size),
                (cx - size, cy),
                (cx - size / 2, cy),
            ]
        draw.polygon(points, fill=fill_color, outline=outline_color, width=line_width)
    else:
        # Filled circle
        x1, y1 = margin, margin
        x2, y2 = img_size - margin, img_size - margin
        draw.ellipse(
            [x1, y1, x2, y2], fill=fill_color, outline=outline_color, width=line_width
        )

    return img


def _icon_nested(
    rng: np.random.Generator,
    img_size: int,
    line_width: int,
    icon_color: tuple[int, int, int],
) -> Image.Image:
    """Nested layered icon with asymmetric marker."""
    img = Image.new("RGB", (img_size, img_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    margin = img_size * 0.1
    colors = [icon_color, (255, 220, 0), (46, 204, 64)]
    scales = [1.0, 0.65, 0.35]

    # Only use non-symmetric shapes (rectangle and triangle, skip circle)
    shape_type = int(rng.integers(0, 2))  # 0=rectangle, 1=triangle

    for scale, color in zip(scales, colors, strict=True):
        m = margin + (1 - scale) * (img_size - 2 * margin) / 2
        x1, y1 = m, m
        x2, y2 = img_size - m, img_size - m

        if shape_type == 0:
            draw.rectangle(
                [x1, y1, x2, y2], fill=color, outline=(40, 40, 40), width=line_width
            )
        else:
            # Triangle (asymmetric)
            cx = img_size / 2
            points = [(cx, y1), (x2, y2), (x1, y2)]
            draw.polygon(points, fill=color, outline=(40, 40, 40), width=line_width)

    # Add asymmetric corner marker to break any remaining symmetry
    marker_x = margin + img_size * 0.1
    marker_y = margin + img_size * 0.1
    marker_size = int(img_size * 0.06)
    draw.ellipse(
        [marker_x, marker_y, marker_x + marker_size, marker_y + marker_size],
        fill=(60, 60, 60),
    )

    return img


def _icon_complex(
    rng: np.random.Generator,
    img_size: int,
    line_width: int,
    icon_color: tuple[int, int, int],
) -> Image.Image:
    """Complex icon (gear, compound shapes)."""
    img = Image.new("RGB", (img_size, img_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    center = img_size // 2
    icon_type = int(rng.integers(0, 3))

    if icon_type == 0:
        # Gear
        outer_r, inner_r = int(img_size * 0.35), int(img_size * 0.2)
        teeth = int(rng.integers(8, 14))
        draw.ellipse(
            (center - inner_r, center - inner_r, center + inner_r, center + inner_r),
            fill=(200, 200, 200),
            outline=(80, 80, 80),
            width=line_width,
        )
        for i in range(teeth):
            angle = i * 2 * np.pi / teeth
            x1 = center + outer_r * np.cos(angle - 0.15)
            y1 = center + outer_r * np.sin(angle - 0.15)
            x2 = center + outer_r * np.cos(angle + 0.15)
            y2 = center + outer_r * np.sin(angle + 0.15)
            x3 = center + inner_r * 1.1 * np.cos(angle + 0.15)
            y3 = center + inner_r * 1.1 * np.sin(angle + 0.15)
            x4 = center + inner_r * 1.1 * np.cos(angle - 0.15)
            y4 = center + inner_r * 1.1 * np.sin(angle - 0.15)
            draw.polygon(
                [(x1, y1), (x2, y2), (x3, y3), (x4, y4)],
                fill=(150, 150, 150),
                outline=(80, 80, 80),
                width=2,
            )
        draw.ellipse(
            (center - 20, center - 20, center + 20, center + 20),
            fill="white",
            outline=(80, 80, 80),
            width=2,
        )
    elif icon_type == 1:
        # Compound: circle + triangle
        margin = img_size * 0.15
        draw.ellipse(
            [margin, margin, img_size - margin, img_size - margin],
            fill=(100, 180, 255),
            outline=(40, 40, 40),
            width=line_width,
        )
        tri_margin = img_size * 0.25
        points = [
            (center, tri_margin),
            (img_size - tri_margin, img_size - tri_margin),
            (tri_margin, img_size - tri_margin),
        ]
        draw.polygon(
            points, fill=(255, 200, 100), outline=(40, 40, 40), width=line_width
        )
    else:
        # Compound: overlapping shapes with asymmetric offset
        colors = [(255, 65, 54), (0, 116, 217), (46, 204, 64)]
        for i in range(3):
            # Use asymmetric offsets to break symmetry
            x_off = i * 20
            y_off = i * 12  # Different from x to break symmetry
            m = img_size * 0.2
            x1, y1 = m + x_off, m + y_off
            size = img_size * 0.45 - i * 15
            if size > 20:
                draw.ellipse(
                    [x1, y1, x1 + size, y1 + size],
                    fill=colors[i],
                    outline=(40, 40, 40),
                    width=2,
                )

    # Add corner marker for extra asymmetry on all icon types
    marker_size = int(img_size * 0.05)
    draw.ellipse(
        [
            img_size * 0.08,
            img_size * 0.08,
            img_size * 0.08 + marker_size,
            img_size * 0.08 + marker_size,
        ],
        fill=(60, 60, 60),
    )

    return img


def generate_random_icon(
    rng: np.random.Generator,
    img_size: int = 200,
    line_width: int = 4,
    icon_color: tuple[int, int, int] | None = None,
    style: str | None = None,
    difficulty: int | None = None,
) -> Image.Image:
    """Generate a random icon-like shape for SymmetryFill.

    Creates shapes similar to the original Sphinx Symmetry Fill task,
    with geometric icons composed of lines, rectangles, and arcs.

    Args:
        rng: numpy random generator for reproducibility
        img_size: Size of the output image (square)
        line_width: Width of the lines
        icon_color: Color of the icon (default: random from palette)
        style: Visual style from ICON_STYLES, or "random" for random.
               If None, uses difficulty to select style.
        difficulty: Difficulty level 1-4. Higher = more complex icons.

    Returns:
        PIL Image with the icon shape
    """
    # Select icon color if not provided
    if icon_color is None:
        icon_colors = [
            (0, 116, 217),  # blue
            (255, 65, 54),  # red
            (46, 204, 64),  # green
            (240, 18, 190),  # pink
            (255, 133, 27),  # orange
            (0, 180, 180),  # cyan
        ]
        icon_color = icon_colors[int(rng.integers(0, len(icon_colors)))]

    # Style selection logic
    if style == "random" or (style is None and difficulty is None):
        selected_style = ICON_STYLES[int(rng.integers(0, len(ICON_STYLES)))]
    elif style is not None and style != "random":
        if style not in ICON_STYLES:
            raise ValueError(f"Unknown style: {style}. Choose from {ICON_STYLES}")
        selected_style = style
    else:
        # Use difficulty to select style
        difficulty = max(1, min(difficulty, len(ICON_STYLES)))
        max_style_idx = difficulty
        selected_style = ICON_STYLES[int(rng.integers(0, max_style_idx))]

    # Dispatch to style-specific function
    style_funcs = {
        "simple": _icon_simple,
        "colored": _icon_colored,
        "nested": _icon_nested,
        "complex": _icon_complex,
    }

    return style_funcs[selected_style](rng, img_size, line_width, icon_color)


def generate_symmetric_2x2_icons(
    rng: np.random.Generator,
    cell_size: int = 200,
    line_width: int = 4,
    style: str | None = None,
    difficulty: int | None = None,
) -> tuple[list[Image.Image], int]:
    """Generate a 2x2 grid of icons with V+H symmetry.

    Similar to generate_symmetric_2x2_grid but uses icon shapes instead of
    colored grids.

    Args:
        rng: numpy random generator for reproducibility
        cell_size: Pixel size of each cell
        line_width: Width of icon lines
        style: Visual style from ICON_STYLES, or "random" for random
        difficulty: Difficulty level 1-4 for style selection

    Returns:
        Tuple of (list of 4 cell images [TL, TR, BL, BR], hidden cell index)
    """
    # Generate the top-left icon
    top_left = generate_random_icon(
        rng,
        img_size=cell_size,
        line_width=line_width,
        style=style,
        difficulty=difficulty,
    )

    # Generate other cells through symmetry transformations
    top_right = apply_transform(top_left, "flip_h")
    bottom_left = apply_transform(top_left, "flip_v")
    bottom_right = apply_transform(top_left, "rot180")

    cells = [top_left, top_right, bottom_left, bottom_right]

    # Randomly select which cell to hide
    hidden_idx = int(rng.integers(0, 4))

    return cells, hidden_idx


def find_bounding_boxes(
    img: Image.Image,
    border_color: tuple[int, int, int] = (0, 0, 0),
    threshold: int = 50,
) -> list[tuple[int, int, int, int]]:
    """Find bounding boxes of regions enclosed by borders.

    This function detects rectangular regions with black borders.

    Args:
        img: Input PIL Image
        border_color: RGB color of the border (default black)
        threshold: Color matching threshold

    Returns:
        List of (x1, y1, x2, y2) bounding boxes
    """
    width, height = img.size
    pixels = img.load()

    def is_border_color(pixel: tuple[int, ...]) -> bool:
        if len(pixel) == 4:  # RGBA
            pixel = pixel[:3]
        return all(abs(pixel[i] - border_color[i]) < threshold for i in range(3))

    # Find horizontal lines (rows with many consecutive border pixels)
    h_lines = []
    for y in range(height):
        line_start = None
        for x in range(width):
            if is_border_color(pixels[x, y]):
                if line_start is None:
                    line_start = x
            else:
                if line_start is not None and x - line_start > 100:
                    h_lines.append((line_start, y, x, y))
                line_start = None
        if line_start is not None and width - line_start > 100:
            h_lines.append((line_start, y, width, y))

    # Find vertical lines
    v_lines = []
    for x in range(width):
        line_start = None
        for y in range(height):
            if is_border_color(pixels[x, y]):
                if line_start is None:
                    line_start = y
            else:
                if line_start is not None and y - line_start > 100:
                    v_lines.append((x, line_start, x, y))
                line_start = None
        if line_start is not None and height - line_start > 100:
            v_lines.append((x, line_start, x, height))

    # Find rectangles by finding intersections
    boxes = []
    for hl1 in h_lines:
        for hl2 in h_lines:
            if hl2[1] <= hl1[1]:
                continue
            for vl1 in v_lines:
                for vl2 in v_lines:
                    if vl2[0] <= vl1[0]:
                        continue
                    # Check if lines form a rectangle
                    x1, y1 = vl1[0], hl1[1]
                    x2, y2 = vl2[0], hl2[1]
                    # Verify corners
                    if (
                        abs(hl1[0] - x1) < 10
                        and abs(hl1[2] - x2) < 10
                        and abs(hl2[0] - x1) < 10
                        and abs(hl2[2] - x2) < 10
                        and abs(vl1[1] - y1) < 10
                        and abs(vl1[3] - y2) < 10
                        and abs(vl2[1] - y1) < 10
                        and abs(vl2[3] - y2) < 10
                    ):
                        if x2 - x1 > 50 and y2 - y1 > 50:
                            boxes.append((x1, y1, x2, y2))

    # Remove duplicates and overlapping boxes
    unique_boxes = []
    for box in boxes:
        is_duplicate = False
        for existing in unique_boxes:
            if (
                abs(box[0] - existing[0]) < 20
                and abs(box[1] - existing[1]) < 20
                and abs(box[2] - existing[2]) < 20
                and abs(box[3] - existing[3]) < 20
            ):
                is_duplicate = True
                break
        if not is_duplicate:
            unique_boxes.append(box)

    return sorted(unique_boxes, key=lambda b: (b[1], b[0]))


def crop_original_shape(img: Image.Image) -> Image.Image:
    """Crop the original shape from the top portion of the image.

    The original shape is typically centered at the top of the image.

    Args:
        img: Original Sphinx image

    Returns:
        Cropped image of the original shape
    """
    width, height = img.size
    boxes = find_bounding_boxes(img)

    if not boxes:
        # Fallback: estimate position based on typical layout
        box_width = min(300, width // 4)
        box_height = min(300, height // 3)
        x1 = (width - box_width) // 2
        y1 = 20
        return img.crop((x1, y1, x1 + box_width, y1 + box_height))

    # Filter out the outer border (boxes that are nearly the full image size)
    inner_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box
        box_width = x2 - x1
        box_height = y2 - y1
        # Skip if box is more than 80% of image size (likely outer border)
        if box_width > width * 0.8 and box_height > height * 0.8:
            continue
        inner_boxes.append(box)

    if not inner_boxes:
        # Fallback
        box_width = min(300, width // 4)
        box_height = min(300, height // 3)
        x1 = (width - box_width) // 2
        y1 = 20
        return img.crop((x1, y1, x1 + box_width, y1 + box_height))

    # The first inner box (topmost, sorted by y then x) should be the original shape
    # Sort by y-coordinate to get the topmost box
    inner_boxes.sort(key=lambda b: (b[1], b[0]))
    x1, y1, x2, y2 = inner_boxes[0]
    # Add small padding inside the border
    return img.crop((x1 + 2, y1 + 2, x2 - 2, y2 - 2))


def crop_options(img: Image.Image) -> list[Image.Image]:
    """Crop the option images from the bottom portion of the image.

    Args:
        img: Original Sphinx image

    Returns:
        List of 4 cropped option images
    """
    boxes = find_bounding_boxes(img)

    # Options are in the bottom row (boxes after the first one)
    if len(boxes) >= 5:
        # First box is original, next 4 are options
        option_boxes = boxes[1:5]
    elif len(boxes) >= 4:
        # Maybe all 4 are options (no detected original)
        option_boxes = boxes[:4]
    else:
        # Fallback: estimate positions
        width, height = img.size
        option_width = width // 5
        option_height = option_width
        y_start = height // 2
        option_boxes = []
        for i in range(4):
            x = (i + 0.5) * (width / 4) - option_width / 2
            option_boxes.append(
                (int(x), y_start, int(x + option_width), y_start + option_height)
            )

    options = []
    for x1, y1, x2, y2 in option_boxes:
        options.append(img.crop((x1 + 2, y1 + 2, x2 - 2, y2 - 2)))

    return options


def compose_8_options(
    original: Image.Image,
    options: list[Image.Image],
    correct_idx: int,
    option_size: int = 280,
    padding: int = 20,
    label_height: int = 50,
) -> Image.Image:
    """Compose a new image with 8 options arranged in 2x4 layout.

    Args:
        original: The original shape image
        options: List of 8 option images
        correct_idx: Index of the correct answer (0-7)
        option_size: Size of each option box
        padding: Padding between elements
        label_height: Height reserved for labels

    Returns:
        Composed image with original at top and 8 options below
    """
    if len(options) != 8:
        raise ValueError(f"Expected 8 options, got {len(options)}")

    # Calculate dimensions
    cols, rows = 4, 2
    total_width = cols * option_size + (cols + 1) * padding
    original_height = option_size + padding * 2
    options_height = rows * (option_size + label_height) + (rows + 1) * padding
    total_height = original_height + options_height

    # Create canvas
    canvas = Image.new("RGB", (total_width, total_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Load font
    assets_dir = resources.files("gym_v.envs") / "assets"
    font_path = assets_dir / "DejaVuSans.ttf"
    try:
        font = ImageFont.truetype(str(font_path), 32)
    except Exception:
        font = ImageFont.load_default()

    # Draw original shape at top center
    orig_resized = original.resize((option_size, option_size), Image.Resampling.LANCZOS)
    orig_x = (total_width - option_size) // 2
    orig_y = padding

    # Draw border for original
    draw.rectangle(
        [orig_x - 2, orig_y - 2, orig_x + option_size + 2, orig_y + option_size + 2],
        outline=(0, 0, 0),
        width=2,
    )
    canvas.paste(orig_resized, (orig_x, orig_y))

    # Draw 8 options in 2x4 grid
    labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]
    for i, opt in enumerate(options):
        row = i // cols
        col = i % cols

        x = padding + col * (option_size + padding)
        y = original_height + padding + row * (option_size + label_height + padding)

        # Resize option
        opt_resized = opt.resize((option_size, option_size), Image.Resampling.LANCZOS)

        # Draw border
        draw.rectangle(
            [x - 2, y - 2, x + option_size + 2, y + option_size + 2],
            outline=(0, 0, 0),
            width=2,
        )
        canvas.paste(opt_resized, (x, y))

        # Draw label
        label = labels[i]
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_x = x + (option_size - text_width) // 2
        text_y = y + option_size + 5
        draw.text((text_x, text_y), label, fill=(0, 0, 0), font=font)

    return canvas


def generate_8_options(
    original: Image.Image,
    correct_transform: str,
    rng: Any,
) -> tuple[list[Image.Image], int]:
    """Generate 8 option images with all transformations.

    Args:
        original: The original shape image
        correct_transform: The correct transformation name
        rng: numpy random generator for shuffling

    Returns:
        Tuple of (list of 8 option images, index of correct answer)
    """
    # Generate all 8 transformations
    transformed = {t: apply_transform(original, t) for t in TRANSFORMS}

    # Create shuffled list
    transform_order = list(TRANSFORMS)
    rng.shuffle(transform_order)

    options = [transformed[t] for t in transform_order]
    correct_idx = transform_order.index(correct_transform)

    return options, correct_idx


def crop_symmetry_fill_parts(
    img: Image.Image,
) -> tuple[Image.Image, list[Image.Image], list[tuple[int, int, int, int]]]:
    """Extract question region and options from Symmetry Fill image.

    Symmetry Fill images have a specific layout:
    - Left side: 2x2 grid with one black (missing) cell
    - Right side: 2x2 grid of options (a)-(d)

    This function uses grid-based extraction which is faster and more reliable
    than bounding box detection for Symmetry Fill images.

    Args:
        img: Original Sphinx Symmetry Fill image

    Returns:
        Tuple of (question_image, list_of_4_option_images, option_boxes)
    """
    width, height = img.size

    # Use grid-based extraction (faster and more reliable for known layout)
    # Left half is question, right half has 2x2 options
    question = img.crop((0, 0, width // 2, height))
    options, option_boxes = _fallback_extract_options(img, width, height)
    return question, options, option_boxes


def _fallback_extract_options(
    img: Image.Image, width: int, height: int
) -> tuple[list[Image.Image], list[tuple[int, int, int, int]]]:
    """Fallback option extraction using grid-based positions.

    Extracts only the icon content, excluding the label at the bottom.
    """
    # Right half has 2x2 options
    right_start = width // 2
    option_width = (width - right_start) // 2
    option_height = height // 2

    option_boxes = [
        (right_start, 0, right_start + option_width, option_height),  # (a)
        (right_start + option_width, 0, width, option_height),  # (b)
        (right_start, option_height, right_start + option_width, height),  # (c)
        (right_start + option_width, option_height, width, height),  # (d)
    ]

    options = []
    for x1, y1, x2, y2 in option_boxes:
        # Add margin to avoid borders
        margin = 5
        # Remove the bottom ~25% which contains the label like "(a)"
        label_ratio = 0.25
        content_height = int((y2 - y1) * (1 - label_ratio))
        opt = img.crop(
            (x1 + margin, y1 + margin, x2 - margin, y1 + content_height - margin)
        )
        options.append(opt)

    return options, option_boxes


def extract_option_content(option_img: Image.Image) -> Image.Image:
    """Extract the icon content from an option image, removing the label.

    Options typically have an icon in a box at the top, and a label like "(a)" at the bottom.

    Args:
        option_img: Single option image with potential label

    Returns:
        Image with just the icon content (label removed)
    """
    width, height = option_img.size

    # The label is typically in the bottom ~15% of the image
    # Crop the top portion that contains the actual icon
    label_ratio = 0.15
    content_height = int(height * (1 - label_ratio))

    return option_img.crop((0, 0, width, content_height))


def compose_symmetry_fill_8_options(
    question: Image.Image,
    options: list[Image.Image],
    correct_idx: int,
    option_size: int = 200,
    padding: int = 15,
    label_height: int = 40,
) -> Image.Image:
    """Compose a new image with question on left and 8 options on right (2x4 layout).

    Args:
        question: The question image (2x2 grid with missing cell)
        options: List of 8 option images
        correct_idx: Index of the correct answer (0-7)
        option_size: Size of each option box
        padding: Padding between elements
        label_height: Height reserved for labels

    Returns:
        Composed image with question on left and 8 options on right
    """
    if len(options) != 8:
        raise ValueError(f"Expected 8 options, got {len(options)}")

    # Calculate dimensions
    cols, rows = 4, 2
    options_width = cols * option_size + (cols + 1) * padding
    options_height = rows * (option_size + label_height) + (rows + 1) * padding

    # Scale question to fit the height
    q_width, q_height = question.size
    q_scale = options_height / q_height
    new_q_width = int(q_width * q_scale)
    new_q_height = int(q_height * q_scale)
    question_resized = question.resize(
        (new_q_width, new_q_height), Image.Resampling.LANCZOS
    )

    # Total canvas size
    total_width = new_q_width + padding + options_width
    total_height = max(new_q_height, options_height)

    # Create canvas
    canvas = Image.new("RGB", (total_width, total_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Load font
    assets_dir = resources.files("gym_v.envs") / "assets"
    font_path = assets_dir / "DejaVuSans.ttf"
    try:
        font = ImageFont.truetype(str(font_path), 28)
    except Exception:
        font = ImageFont.load_default()

    # Paste question on left
    q_y = (total_height - new_q_height) // 2
    canvas.paste(question_resized, (0, q_y))

    # Draw 8 options in 2x4 grid on the right
    labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]
    options_start_x = new_q_width + padding

    for i, opt in enumerate(options):
        row = i // cols
        col = i % cols

        x = options_start_x + padding + col * (option_size + padding)
        y = padding + row * (option_size + label_height + padding)

        # Resize option to fit
        opt_resized = opt.resize((option_size, option_size), Image.Resampling.LANCZOS)

        # Draw border
        draw.rectangle(
            [x - 2, y - 2, x + option_size + 2, y + option_size + 2],
            outline=(0, 0, 0),
            width=2,
        )
        canvas.paste(opt_resized, (x, y))

        # Draw label
        label = labels[i]
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_x = x + (option_size - text_width) // 2
        text_y = y + option_size + 5
        draw.text((text_x, text_y), label, fill=(0, 0, 0), font=font)

    return canvas


def _images_are_similar(
    img1: Image.Image, img2: Image.Image, threshold: float = 10.0
) -> bool:
    """Check if two images are visually similar.

    Args:
        img1: First image
        img2: Second image
        threshold: Mean absolute difference threshold (lower = more similar)

    Returns:
        True if images are similar, False otherwise
    """
    import numpy as np

    size = (100, 100)
    arr1 = np.array(img1.resize(size))
    arr2 = np.array(img2.resize(size))

    if arr1.shape != arr2.shape:
        return False

    diff = np.mean(np.abs(arr1.astype(float) - arr2.astype(float)))
    return diff < threshold


def _add_asymmetric_marker(
    img: Image.Image,
    rng: Any,
    marker_size: int = 8,
) -> Image.Image:
    """Add a small asymmetric marker to ensure image is not rotationally symmetric.

    Args:
        img: Input image
        rng: Random number generator
        marker_size: Size of the marker

    Returns:
        Image with asymmetric marker added
    """
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)

    # Add a small colored dot in a random corner region (but not center)
    w, h = img_copy.size
    margin = int(w * 0.15)

    # Choose one of the corner regions randomly
    corner = int(rng.integers(0, 4))
    if corner == 0:  # Top-left
        x = margin + int(rng.integers(0, margin))
        y = margin + int(rng.integers(0, margin))
    elif corner == 1:  # Top-right
        x = w - margin - int(rng.integers(0, margin))
        y = margin + int(rng.integers(0, margin))
    elif corner == 2:  # Bottom-left
        x = margin + int(rng.integers(0, margin))
        y = h - margin - int(rng.integers(0, margin))
    else:  # Bottom-right
        x = w - margin - int(rng.integers(0, margin))
        y = h - margin - int(rng.integers(0, margin))

    # Draw a small filled circle as marker
    marker_color = (80, 80, 80)
    draw.ellipse(
        [
            x - marker_size // 2,
            y - marker_size // 2,
            x + marker_size // 2,
            y + marker_size // 2,
        ],
        fill=marker_color,
    )

    return img_copy


def generate_extra_distractors(
    correct_option: Image.Image,
    original_options: list[Image.Image],
    rng: Any,
    num_extra: int = 4,
) -> list[Image.Image]:
    """Generate additional distractor options by applying transforms to the correct answer.

    Ensures no duplicate options by checking similarity against both original options
    and already-generated distractors.

    Args:
        correct_option: The correct answer option image
        original_options: List of original 4 options (for comparison to avoid duplicates)
        rng: numpy random generator
        num_extra: Number of extra distractors to generate

    Returns:
        List of extra distractor images
    """
    # All images to compare against (original options + distractors we generate)
    all_existing = list(original_options)

    # Apply all transforms to the correct option
    all_transforms = list(TRANSFORMS)
    rng.shuffle(all_transforms)

    distractors = []
    for transform in all_transforms:
        if len(distractors) >= num_extra:
            break

        # Skip identity transform (would be same as correct answer)
        if transform == "identity":
            continue

        transformed = apply_transform(correct_option, transform)

        # Check if this transform is too similar to any existing option
        is_duplicate = any(
            _images_are_similar(transformed, existing) for existing in all_existing
        )

        if not is_duplicate:
            distractors.append(transformed)
            all_existing.append(transformed)

    # If we still don't have enough (due to symmetric shapes), generate variants
    # by applying transforms to OTHER original options
    if len(distractors) < num_extra:
        for orig in original_options:
            if len(distractors) >= num_extra:
                break
            for transform in all_transforms:
                if len(distractors) >= num_extra:
                    break
                if transform == "identity":
                    continue

                transformed = apply_transform(orig, transform)
                is_duplicate = any(
                    _images_are_similar(transformed, existing)
                    for existing in all_existing
                )

                if not is_duplicate:
                    distractors.append(transformed)
                    all_existing.append(transformed)

    # Last resort: add markers to make unique versions
    while len(distractors) < num_extra:
        # Create a variation by adding asymmetric marker to a transform
        transform = all_transforms[len(distractors) % len(all_transforms)]
        transformed = apply_transform(correct_option, transform)
        marked = _add_asymmetric_marker(transformed, rng)
        distractors.append(marked)

    return distractors[:num_extra]


# Sequence patterns for SequenceCompletion task
SEQUENCE_PATTERNS = {
    "rot90_sequence": ["identity", "rot90_cw", "rot180", "rot90_ccw"],
    "flip_h_sequence": ["identity", "flip_h"],
    "flip_v_sequence": ["identity", "flip_v"],
    "rot180_sequence": ["identity", "rot180"],
    "diagonal_flip": ["identity", "flip_diag", "rot180", "flip_antidiag"],
}


def compose_odd_one_out_8_options(
    options: list[Image.Image],
    odd_idx: int,
    option_size: int = 200,
    padding: int = 15,
    label_height: int = 40,
) -> Image.Image:
    """Compose 8 options in 2x4 layout for OddOneOut task (no original at top).

    Args:
        options: List of 8 option images
        odd_idx: Index of the odd one out (for reference, not displayed)
        option_size: Size of each option box
        padding: Padding between elements
        label_height: Height reserved for labels

    Returns:
        Composed image with 8 options in 2x4 grid
    """
    if len(options) != 8:
        raise ValueError(f"Expected 8 options, got {len(options)}")

    cols, rows = 4, 2
    total_width = cols * option_size + (cols + 1) * padding
    total_height = rows * (option_size + label_height) + (rows + 1) * padding

    canvas = Image.new("RGB", (total_width, total_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    assets_dir = resources.files("gym_v.envs") / "assets"
    font_path = assets_dir / "DejaVuSans.ttf"
    try:
        font = ImageFont.truetype(str(font_path), 28)
    except Exception:
        font = ImageFont.load_default()

    labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]

    for i, opt in enumerate(options):
        row = i // cols
        col = i % cols

        x = padding + col * (option_size + padding)
        y = padding + row * (option_size + label_height + padding)

        opt_resized = opt.resize((option_size, option_size), Image.Resampling.LANCZOS)

        draw.rectangle(
            [x - 2, y - 2, x + option_size + 2, y + option_size + 2],
            outline=(0, 0, 0),
            width=2,
        )
        canvas.paste(opt_resized, (x, y))

        label = labels[i]
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_x = x + (option_size - text_width) // 2
        text_y = y + option_size + 5
        draw.text((text_x, text_y), label, fill=(0, 0, 0), font=font)

    return canvas


def compose_sequence_completion_image(
    sequence: list[Image.Image],
    options: list[Image.Image],
    correct_idx: int,
    option_size: int = 150,
    padding: int = 10,
    label_height: int = 35,
) -> Image.Image:
    """Compose image with sequence at top and 8 options below.

    Layout:
    - Top row: Sequence items followed by "?" placeholder
    - Bottom: 2x4 grid of options (a)-(h)

    Args:
        sequence: List of sequence images (shown items)
        options: List of 8 option images
        correct_idx: Index of the correct answer
        option_size: Size of each option/sequence box
        padding: Padding between elements
        label_height: Height reserved for labels

    Returns:
        Composed image with sequence and options
    """
    if len(options) != 8:
        raise ValueError(f"Expected 8 options, got {len(options)}")

    cols, rows = 4, 2
    sequence_count = len(sequence) + 1  # +1 for "?" placeholder

    # Calculate dimensions
    sequence_width = sequence_count * option_size + (sequence_count + 1) * padding
    options_width = cols * option_size + (cols + 1) * padding
    total_width = max(sequence_width, options_width)

    sequence_height = option_size + padding * 2
    options_height = rows * (option_size + label_height) + (rows + 1) * padding
    total_height = sequence_height + options_height

    canvas = Image.new("RGB", (total_width, total_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    assets_dir = resources.files("gym_v.envs") / "assets"
    font_path = assets_dir / "DejaVuSans.ttf"
    try:
        font = ImageFont.truetype(str(font_path), 28)
        large_font = ImageFont.truetype(str(font_path), 48)
    except Exception:
        font = ImageFont.load_default()
        large_font = font

    # Draw sequence at top (centered)
    seq_start_x = (total_width - sequence_width) // 2 + padding
    seq_y = padding

    for i, seq_img in enumerate(sequence):
        x = seq_start_x + i * (option_size + padding)
        seq_resized = seq_img.resize(
            (option_size, option_size), Image.Resampling.LANCZOS
        )
        draw.rectangle(
            [x - 2, seq_y - 2, x + option_size + 2, seq_y + option_size + 2],
            outline=(0, 0, 0),
            width=2,
        )
        canvas.paste(seq_resized, (x, seq_y))

    # Draw "?" placeholder
    q_x = seq_start_x + len(sequence) * (option_size + padding)
    draw.rectangle(
        [q_x - 2, seq_y - 2, q_x + option_size + 2, seq_y + option_size + 2],
        outline=(100, 100, 100),
        width=2,
    )
    # Draw "?" in center
    q_bbox = draw.textbbox((0, 0), "?", font=large_font)
    q_text_w = q_bbox[2] - q_bbox[0]
    q_text_h = q_bbox[3] - q_bbox[1]
    q_text_x = q_x + (option_size - q_text_w) // 2
    q_text_y = seq_y + (option_size - q_text_h) // 2
    draw.text((q_text_x, q_text_y), "?", fill=(100, 100, 100), font=large_font)

    # Draw 8 options below
    labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]
    options_start_x = (total_width - options_width) // 2 + padding
    options_start_y = sequence_height

    for i, opt in enumerate(options):
        row = i // cols
        col = i % cols

        x = options_start_x + col * (option_size + padding)
        y = options_start_y + padding + row * (option_size + label_height + padding)

        opt_resized = opt.resize((option_size, option_size), Image.Resampling.LANCZOS)

        draw.rectangle(
            [x - 2, y - 2, x + option_size + 2, y + option_size + 2],
            outline=(0, 0, 0),
            width=2,
        )
        canvas.paste(opt_resized, (x, y))

        label = labels[i]
        bbox = draw.textbbox((0, 0), label, font=font)
        text_width = bbox[2] - bbox[0]
        text_x = x + (option_size - text_width) // 2
        text_y = y + option_size + 5
        draw.text((text_x, text_y), label, fill=(0, 0, 0), font=font)

    return canvas
