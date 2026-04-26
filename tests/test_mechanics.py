from pathlib import Path

from baba_rlvr.engine import Direction, parse_level
from baba_rlvr.levels.map_writer import parse_grid, read_map, write_map


def _level(tmp_path: Path, name: str, grid: str):
    path = write_map(tmp_path / f"{name}.txt", parse_grid(grid))
    return parse_level({"map_path": str(path), "max_steps": 30}), path


def test_water_is_visible_and_round_trips(tmp_path: Path):
    world, path = _level(
        tmp_path,
        "water",
        """
        wall wall wall wall wall
        wall BABA IS YOU wall
        wall WATER IS SINK wall
        wall baba water flag wall
        wall FLAG IS WIN wall
        wall wall wall wall wall
        """,
    )

    assert "water" in world.to_tokens()[3][2]
    assert read_map(path)[2][1:4] == ["WATER", "IS", "SINK"]
    assert ("water", "SINK") in {(e.value, p.value) for e, p in world.rules}


def test_and_expands_property_rules(tmp_path: Path):
    world, _path = _level(
        tmp_path,
        "and",
        """
        wall wall wall wall wall wall wall wall
        wall BABA IS YOU AND SINK . wall
        wall JELLY IS SINK . . . wall
        wall baba jelly flag . . . wall
        wall FLAG IS WIN . . . wall
        wall wall wall wall wall wall wall wall
        """,
    )

    rules = {(e.value, p.value) for e, p in world.rules}
    assert ("baba", "YOU") in rules
    assert ("baba", "SINK") in rules
    assert ("jelly", "SINK") in rules


def test_sink_removes_mover_and_sinker(tmp_path: Path):
    world, _path = _level(
        tmp_path,
        "sink",
        """
        wall wall wall wall wall wall wall
        wall BABA IS YOU . . wall
        wall WATER IS SINK . . wall
        wall baba water flag . . wall
        wall FLAG IS WIN . . wall
        wall wall wall wall wall wall wall
        """,
    )

    info = world.step(Direction.RIGHT)

    assert info["died"]
    assert world.lost
    assert world.to_tokens()[3][1:4] == [".", ".", "flag"]


def test_and_open_shut_push_chain(tmp_path: Path):
    world, _path = _level(
        tmp_path,
        "open_shut",
        """
        wall wall wall wall wall wall wall wall
        wall KEKE IS YOU . . . wall
        wall KEY IS PUSH AND OPEN . wall
        wall DOOR IS SHUT . . . wall
        wall HEDGE AND DOOR IS STOP . wall
        wall keke key door flag . . wall
        wall FLAG IS WIN . . . wall
        wall wall wall wall wall wall wall wall
        """,
    )
    rules = {(e.value, p.value) for e, p in world.rules}
    assert ("key", "PUSH") in rules
    assert ("key", "OPEN") in rules
    assert ("door", "SHUT") in rules
    assert ("door", "STOP") in rules
    assert ("hedge", "STOP") in rules

    world.step(Direction.RIGHT)

    assert world.to_tokens()[5][1:5] == [".", "keke", ".", "flag"]

    world.step(Direction.RIGHT)
    world.step(Direction.RIGHT)
    assert world.won


def test_text_push_chain_does_not_spawn_subject_icon(tmp_path: Path):
    world, _path = _level(
        tmp_path,
        "text_push_chain",
        """
        wall wall wall wall wall wall wall
        wall BABA IS YOU . . wall
        wall baba ROCK IS PUSH . wall
        wall wall wall wall wall wall wall
        """,
    )

    world.step(Direction.RIGHT)

    assert world.to_tokens()[2] == ["wall", ".", "baba", "ROCK", "IS", "PUSH", "wall"]
    assert all("rock" not in cell.split(",") for cell in world.to_tokens()[2])


def test_push_moves_only_objects_present_on_target_tile(tmp_path: Path):
    world, _path = _level(
        tmp_path,
        "push_only_present",
        """
        wall wall wall wall wall wall wall
        wall BABA IS YOU . . wall
        wall ROCK IS PUSH . . wall
        wall KEY IS PUSH . . wall
        wall baba rock . . . wall
        wall wall wall wall wall wall wall
        """,
    )

    world.step(Direction.RIGHT)

    assert world.to_tokens()[4] == ["wall", ".", "baba", "rock", ".", ".", "wall"]
    assert all("key" not in cell.split(",") for row in world.to_tokens() for cell in row)
