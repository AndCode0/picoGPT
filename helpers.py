import json
from pathlib import Path

def save_vocab(vocab: dict[int, bytes], path: Path) -> None:
    serialized = {str(idx): token.hex() for idx, token in vocab.items()}
    path.write_text(json.dumps(serialized, indent=2))

def load_vocab(path: Path) -> dict[int, bytes]:
    raw = json.loads(path.read_text())
    return {int(idx): bytes.fromhex(hex_str) for idx, hex_str in raw.items()}

def save_merges(merges: list[tuple[bytes, bytes]], path: Path) -> None:
    lines = [f"{a.hex()} {b.hex()}" for a, b in merges]
    path.write_text("\n".join(lines))

def load_merges(path: Path) -> list[tuple[bytes, bytes]]:
    lines = path.read_text().strip().split("\n")
    return [
        (bytes.fromhex(parts[0]), bytes.fromhex(parts[1]))
        for line in lines
        if (parts:= line.split(" ")) and len(parts) == 2
    ]