"""
Custom 64x64 RGB renderer for FrozenLake-v1.
Produces consistent, visually clear grid images for world-model training.
"""

import numpy as np
from PIL import Image, ImageDraw


# Cell color palette (RGB)
COLORS = {
    "S": (180, 220, 255),   # Start (frozen) - light blue
    "F": (180, 220, 255),   # Frozen - light blue
    "H": ( 25,  25,  60),   # Hole - near black-blue
    "G": ( 60, 200,  80),   # Goal - green
}
AGENT_COLOR   = (230,  80,  30)   # Orange-red
GRID_COLOR    = ( 90, 130, 180)   # Grid line color
IMG_SIZE      = 64
GRID_LINES    = True


def render_state(map_desc, state: int, img_size: int = IMG_SIZE) -> np.ndarray:
    """
    Render a FrozenLake state as an RGB image.

    Args:
        map_desc: list of strings describing the map, e.g. ["SFFF", "FHFH", ...]
        state:    integer state index (row * ncols + col)
        img_size: output image size (square)

    Returns:
        uint8 numpy array of shape (img_size, img_size, 3)
    """
    nrows = len(map_desc)
    ncols = len(map_desc[0])
    cell_size = img_size // ncols  # pixels per cell

    img = Image.new("RGB", (img_size, img_size), color=(200, 200, 200))
    draw = ImageDraw.Draw(img)

    agent_row = state // ncols
    agent_col = state % ncols

    for r in range(nrows):
        for c in range(ncols):
            cell = map_desc[r][c]
            color = COLORS.get(cell, COLORS["F"])

            x0 = c * cell_size
            y0 = r * cell_size
            x1 = x0 + cell_size - 1
            y1 = y0 + cell_size - 1

            # Fill cell
            draw.rectangle([x0, y0, x1, y1], fill=color)

            # Draw goal marker (star-like inner highlight)
            if cell == "G":
                margin = cell_size // 4
                draw.rectangle(
                    [x0 + margin, y0 + margin, x1 - margin, y1 - margin],
                    fill=(255, 230, 50)
                )

            # Draw hole marker (inner dark circle)
            if cell == "H":
                margin = cell_size // 5
                draw.ellipse(
                    [x0 + margin, y0 + margin, x1 - margin, y1 - margin],
                    fill=(10, 10, 40)
                )

    # Draw grid lines
    if GRID_LINES:
        for i in range(ncols + 1):
            x = i * cell_size
            draw.line([(x, 0), (x, img_size - 1)], fill=GRID_COLOR, width=1)
        for i in range(nrows + 1):
            y = i * cell_size
            draw.line([(0, y), (img_size - 1, y)], fill=GRID_COLOR, width=1)

    # Draw agent (filled circle)
    ar, ac = agent_row, agent_col
    cx = ac * cell_size + cell_size // 2
    cy = ar * cell_size + cell_size // 2
    radius = cell_size // 3
    draw.ellipse(
        [cx - radius, cy - radius, cx + radius, cy + radius],
        fill=AGENT_COLOR, outline=(255, 255, 255), width=1
    )

    return np.array(img, dtype=np.uint8)


def map_desc_to_list(map_desc) -> list:
    """Convert map descriptor to list of strings."""
    if isinstance(map_desc, list) and isinstance(map_desc[0], bytes):
        return [row.decode("utf-8") for row in map_desc]
    return list(map_desc)


if __name__ == "__main__":
    import gymnasium as gym
    import matplotlib.pyplot as plt

    env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=False)
    map_desc = map_desc_to_list(env.unwrapped.desc)
    print("Map:")
    for row in map_desc:
        print(" ", row)

    obs, _ = env.reset()
    img = render_state(map_desc, obs)
    print(f"Rendered image shape: {img.shape}, dtype: {img.dtype}")

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for ax, state in zip(axes, [0, 10, 30, 63]):
        ax.imshow(render_state(map_desc, state))
        ax.set_title(f"State {state}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("/data/koe/ECE285-Final/results/figures/sample_renders.png", dpi=100)
    print("Saved sample renders.")
