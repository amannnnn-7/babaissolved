from baba_rlvr.levels.map_writer import parse_grid, read_map, tokenize, write_map


def test_write_read_map_round_trip(tmp_path):
    rows = parse_grid(
        """
        wall wall wall wall wall
        wall BABA IS YOU wall
        wall baba . flag wall
        wall FLAG IS WIN wall
        wall wall wall wall wall
        """
    )

    path = write_map(tmp_path / "roundtrip.txt", rows)

    assert read_map(path) == rows
    assert tokenize(".") == tokenize("")

