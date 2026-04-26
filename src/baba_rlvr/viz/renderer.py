"""PIL-based renderer for Baba worlds and trajectories.

Design choices
--------------
* Prefer the vendored baba-is-auto sprites when available, with a shape fallback
  so rendering still works in headless/minimal environments.
* Each entity has a glyph + base color; each text block is drawn as a
  rounded rectangle with the word inside, color-coded by category
  (noun=peach, verb=white, property=cyan/red).
* Active rules + the YOU/WIN sets are drawn in a sidebar so the viewer can
  literally watch the rule set change as the agent rewrites the world.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from ..engine import Direction, EntityKind, Tile, WordKind, World
from ..engine.types import NOUN_WORDS, PROPERTY_WORDS, VERB_WORDS

# ----------------------------------------------------------------------------
# Color palette (RGB). Kept colorblind-friendly and high-contrast.
# ----------------------------------------------------------------------------
BG_COLOR = (24, 24, 28)
GRID_COLOR = (60, 60, 70)
SIDEBAR_BG = (16, 16, 20)
TEXT_COLOR = (235, 235, 240)
DIM_TEXT = (140, 140, 150)

ENTITY_COLORS: dict[EntityKind, tuple[int, int, int]] = {
    EntityKind.BABA: (245, 245, 245),   # white
    EntityKind.ROCK: (170, 110, 60),    # brown
    EntityKind.WALL: (95, 95, 110),     # gray
    EntityKind.FLAG: (250, 215, 60),    # gold
    EntityKind.SKULL: (200, 60, 60),    # red
    EntityKind.LAVA: (240, 100, 30),    # orange
    EntityKind.KEKE: (240, 120, 200),   # pink
    EntityKind.DOOR: (140, 90, 200),    # purple
    EntityKind.KEY: (250, 235, 150),    # pale gold
    EntityKind.WATER: (60, 130, 230),
    EntityKind.GRASS: (80, 180, 90),
    EntityKind.TILE: (120, 120, 130),
    EntityKind.FLOWER: (250, 130, 210),
    EntityKind.ICE: (160, 230, 255),
    EntityKind.JELLY: (180, 120, 240),
    EntityKind.CRAB: (230, 80, 70),
    EntityKind.LOVE: (245, 80, 130),
    EntityKind.ALGAE: (40, 150, 110),
    EntityKind.HEDGE: (40, 120, 60),
    EntityKind.BELT: (90, 90, 100),
    EntityKind.BUG: (120, 190, 80),
    EntityKind.ROBOT: (170, 180, 200),
    EntityKind.STAR: (255, 230, 80),
}

ENTITY_GLYPHS: dict[EntityKind, str] = {
    EntityKind.BABA: "B",
    EntityKind.ROCK: "R",
    EntityKind.WALL: "#",
    EntityKind.FLAG: "F",
    EntityKind.SKULL: "S",
    EntityKind.LAVA: "~",
    EntityKind.KEKE: "K",
    EntityKind.DOOR: "D",
    EntityKind.KEY: "k",
    EntityKind.WATER: "~",
    EntityKind.GRASS: "'",
    EntityKind.TILE: "_",
    EntityKind.FLOWER: "*",
    EntityKind.ICE: "I",
    EntityKind.JELLY: "J",
    EntityKind.CRAB: "C",
    EntityKind.LOVE: "L",
    EntityKind.ALGAE: "A",
    EntityKind.HEDGE: "H",
    EntityKind.BELT: "=",
    EntityKind.BUG: "G",
    EntityKind.ROBOT: "Rb",
    EntityKind.STAR: "*",
}

NOUN_BG = (250, 200, 160)   # peach
VERB_BG = (235, 235, 240)   # near-white
PROP_BG = (130, 200, 230)   # cyan
_SPRITES_DIR = (
    Path(__file__).resolve().parents[3]
    / "vendor"
    / "baba-is-auto"
    / "Extensions"
    / "BabaGUI"
    / "sprites"
)
_SPRITE_CACHE: dict[tuple[str, int], Image.Image | None] = {}


def _word_bg(w: WordKind) -> tuple[int, int, int]:
    if w in NOUN_WORDS:
        return NOUN_BG
    if w in VERB_WORDS:
        return VERB_BG
    if w in PROPERTY_WORDS:
        return PROP_BG
    return (200, 200, 200)


# ----------------------------------------------------------------------------
# Font helpers
# ----------------------------------------------------------------------------
def _font(size: int) -> ImageFont.ImageFont:
    """Best-effort: try a TrueType, fall back to PIL default."""
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


# ----------------------------------------------------------------------------
# Tile rendering
# ----------------------------------------------------------------------------
def _draw_tile(
    draw: ImageDraw.ImageDraw, tile: Tile, x: int, y: int, cell: int
) -> None:
    if tile.is_empty:
        return

    # Words take precedence visually (they're the rules!).
    if tile.words:
        w = tile.words[0]
        bg = _word_bg(w)
        pad = max(2, cell // 12)
        draw.rounded_rectangle(
            [x + pad, y + pad, x + cell - pad, y + cell - pad],
            radius=max(3, cell // 8),
            fill=bg,
            outline=(40, 40, 40),
            width=1,
        )
        # Fit the text — use a font that scales with cell size.
        label = w.value
        font = _font(max(8, cell // (max(3, len(label) // 2))))
        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(
            (x + (cell - tw) / 2 - bbox[0], y + (cell - th) / 2 - bbox[1]),
            label,
            fill=(20, 20, 20),
            font=font,
        )
        return

    # Otherwise render the topmost entity.
    e = tile.entities[0]
    color = ENTITY_COLORS.get(e, (200, 200, 200))

    if e == EntityKind.WALL:
        draw.rectangle([x + 1, y + 1, x + cell - 1, y + cell - 1], fill=color)
    elif e in {EntityKind.LAVA, EntityKind.WATER}:
        draw.rectangle([x + 1, y + 1, x + cell - 1, y + cell - 1], fill=color)
        draw.line([x, y + cell // 2, x + cell, y + cell // 2], fill=(255, 200, 100), width=1)
    else:
        # Rounded blob with the entity's glyph in contrasting color.
        pad = max(3, cell // 8)
        draw.rounded_rectangle(
            [x + pad, y + pad, x + cell - pad, y + cell - pad],
            radius=max(3, cell // 5),
            fill=color,
        )
        glyph = ENTITY_GLYPHS.get(e, "?")
        font = _font(max(10, cell // 2))
        bbox = draw.textbbox((0, 0), glyph, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(
            (x + (cell - tw) / 2 - bbox[0], y + (cell - th) / 2 - bbox[1]),
            glyph,
            fill=(20, 20, 24),
            font=font,
        )

    # If multiple entities share the tile, draw a small dot per extra.
    for i, _ in enumerate(tile.entities[1:], start=1):
        cx = x + cell - 4 - 5 * i
        cy = y + cell - 6
        draw.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill=(255, 255, 255))


def _sprite_path(tile: Tile) -> Path | None:
    if tile.words:
        return _SPRITES_DIR / "text" / f"{tile.words[0].value}.gif"
    if tile.entities:
        return _SPRITES_DIR / "icon" / f"{tile.entities[0].value.upper()}.gif"
    return None


def _load_sprite(path: Path, cell: int) -> Image.Image | None:
    key = (str(path), cell)
    if key in _SPRITE_CACHE:
        return _SPRITE_CACHE[key]
    if not path.exists():
        _SPRITE_CACHE[key] = None
        return None
    try:
        img = Image.open(path).convert("RGBA").resize((cell, cell), Image.Resampling.NEAREST)
    except OSError:
        _SPRITE_CACHE[key] = None
        return None
    _SPRITE_CACHE[key] = img
    return img


def _draw_sprite_tile(img: Image.Image, tile: Tile, x: int, y: int, cell: int) -> bool:
    path = _sprite_path(tile)
    if path is None:
        return True
    sprite = _load_sprite(path, cell)
    if sprite is None:
        return False
    img.alpha_composite(sprite, (x, y))
    return True


# ----------------------------------------------------------------------------
# Public: render a single world
# ----------------------------------------------------------------------------
def render_world(
    world: World,
    *,
    cell: int = 48,
    sidebar_w: int = 260,
    title: str = "",
    action_taken: str | None = None,
    reward: float | None = None,
    step_idx: int | None = None,
    backend: str = "sprites",
) -> Image.Image:
    """Render a single world state to a PIL Image with rule sidebar."""
    if backend not in {"sprites", "shapes"}:
        raise ValueError(f"unknown render backend: {backend}")
    grid_w = world.width * cell
    grid_h = world.height * cell
    pad = 16
    img_w = grid_w + sidebar_w + pad * 3
    img_h = max(grid_h, 360) + pad * 2 + 30
    img = Image.new("RGBA", (img_w, img_h), (*BG_COLOR, 255))
    draw = ImageDraw.Draw(img)

    # Title bar
    title_font = _font(16)
    if title:
        draw.text((pad, pad // 2), title, fill=TEXT_COLOR, font=title_font)

    # Grid backdrop
    gx, gy = pad, pad + 24
    draw.rectangle([gx, gy, gx + grid_w, gy + grid_h], fill=(34, 34, 40))
    for r in range(world.height + 1):
        draw.line([gx, gy + r * cell, gx + grid_w, gy + r * cell], fill=GRID_COLOR)
    for c in range(world.width + 1):
        draw.line([gx + c * cell, gy, gx + c * cell, gy + grid_h], fill=GRID_COLOR)
    for y in range(world.height):
        for x in range(world.width):
            tile = world.grid[y][x]
            tx, ty = gx + x * cell, gy + y * cell
            if backend == "sprites" and _draw_sprite_tile(img, tile, tx, ty, cell):
                continue
            _draw_tile(draw, tile, tx, ty, cell)

    # Sidebar
    sx = gx + grid_w + pad
    sy = gy
    draw.rectangle([sx, sy, sx + sidebar_w, sy + grid_h], fill=SIDEBAR_BG)
    f_h = _font(13)
    f_b = _font(14)
    f_lbl = _font(11)

    cy = sy + 8
    if step_idx is not None:
        draw.text((sx + 8, cy), f"step {step_idx} / {world.max_steps}", fill=DIM_TEXT, font=f_lbl)
        cy += 16
    if action_taken is not None:
        draw.text((sx + 8, cy), f"action: {action_taken}", fill=TEXT_COLOR, font=f_b)
        cy += 18
    if reward is not None:
        col = (140, 230, 140) if reward >= 0 else (240, 130, 130)
        draw.text((sx + 8, cy), f"reward: {reward:+.2f}", fill=col, font=f_b)
        cy += 18

    cy += 6
    draw.text((sx + 8, cy), "YOU", fill=DIM_TEXT, font=f_lbl)
    cy += 14
    draw.text(
        (sx + 8, cy),
        ", ".join(sorted(e.value for e in world.you_entities())) or "(none)",
        fill=(245, 245, 245),
        font=f_h,
    )
    cy += 22

    draw.text((sx + 8, cy), "WIN", fill=DIM_TEXT, font=f_lbl)
    cy += 14
    draw.text(
        (sx + 8, cy),
        ", ".join(sorted(e.value for e in world.win_entities())) or "(none)",
        fill=(245, 235, 130),
        font=f_h,
    )
    cy += 22

    draw.text((sx + 8, cy), "ACTIVE RULES", fill=DIM_TEXT, font=f_lbl)
    cy += 14
    rules = sorted(world.rules)
    if not rules:
        draw.text((sx + 8, cy), "(none)", fill=DIM_TEXT, font=f_h)
    for e, p in rules:
        if cy > sy + grid_h - 16:
            break
        draw.text((sx + 8, cy), f"• {e.value} IS {p.value}", fill=TEXT_COLOR, font=f_h)
        cy += 16

    if world.won:
        _draw_banner(draw, img_w, img_h, "WIN!", (60, 200, 100))
    elif world.lost:
        _draw_banner(draw, img_w, img_h, "LOST", (220, 80, 80))

    return img.convert("RGB")


def _draw_banner(draw: ImageDraw.ImageDraw, w: int, h: int, text: str, color) -> None:
    f = _font(36)
    bbox = draw.textbbox((0, 0), text, font=f)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    bx, by = (w - tw) // 2 - bbox[0], (h - th) // 2 - bbox[1]
    pad = 16
    draw.rounded_rectangle(
        [bx - pad, by - pad // 2, bx + tw + pad, by + th + pad // 2],
        radius=12,
        fill=color,
    )
    draw.text((bx, by), text, fill=(20, 20, 20), font=f)


# ----------------------------------------------------------------------------
# Trajectory helpers
# ----------------------------------------------------------------------------
def actions_from_strings(seq: Iterable[str]) -> list[Direction]:
    """Convert ['u','r','r','d'] or ['up','right',...] to Directions.

    Single-character shortcuts: u/d/l/r/w. Anything else is parsed via
    Direction(value).
    """
    short = {"u": Direction.UP, "d": Direction.DOWN, "l": Direction.LEFT,
             "r": Direction.RIGHT, "w": Direction.WAIT}
    out: list[Direction] = []
    for s in seq:
        s = s.strip().lower()
        if not s:
            continue
        if s in short:
            out.append(short[s])
        else:
            out.append(Direction(s))
    return out


def rollout_actions(
    world: World, actions: Iterable[Direction]
) -> list[tuple[World, Direction | None, dict]]:
    """Step `world` through `actions`, returning a list of frame snapshots.

    The first frame is (initial_world, None, {}) — useful for the visualizer
    to show the starting state before any move.
    """
    frames: list[tuple[World, Direction | None, dict]] = [(world.clone(), None, {})]
    cur = world.clone()
    for a in actions:
        info = cur.step(a)
        frames.append((cur.clone(), a, info))
        if cur.won or cur.lost:
            break
    return frames


def render_trajectory_gif(
    frames: list[tuple[World, Direction | None, dict]],
    out: str | Path,
    *,
    cell: int = 48,
    title: str = "",
    rewards: list[float] | None = None,
    duration_ms: int = 400,
    backend: str = "sprites",
) -> Path:
    """Save an animated GIF of a trajectory."""
    images: list[Image.Image] = []
    for i, (w, a, _info) in enumerate(frames):
        r = rewards[i - 1] if (rewards is not None and i > 0 and i - 1 < len(rewards)) else None
        images.append(
            render_world(
                w,
                cell=cell,
                title=title,
                action_taken=a.value if a else None,
                reward=r,
                step_idx=i,
                backend=backend,
            )
        )
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        out,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    return out


def render_trajectory_strip(
    frames: list[tuple[World, Direction | None, dict]],
    out: str | Path,
    *,
    cell: int = 36,
    cols: int = 6,
    title: str = "",
    backend: str = "sprites",
) -> Path:
    """Save a single PNG with all frames laid out in a grid (good for blogs)."""
    panels = [
        render_world(
            w,
            cell=cell,
            sidebar_w=180,
            title="",
            action_taken=a.value if a else None,
            step_idx=i,
            backend=backend,
        )
        for i, (w, a, _info) in enumerate(frames)
    ]
    pw, ph = panels[0].size
    rows = (len(panels) + cols - 1) // cols
    title_h = 32 if title else 8
    canvas = Image.new("RGB", (pw * cols, ph * rows + title_h), BG_COLOR)
    if title:
        ImageDraw.Draw(canvas).text((10, 8), title, fill=TEXT_COLOR, font=_font(18))
    for i, p in enumerate(panels):
        r, c = divmod(i, cols)
        canvas.paste(p, (c * pw, title_h + r * ph))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    return out
