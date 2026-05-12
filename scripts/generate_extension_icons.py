#!/usr/bin/env python3
from __future__ import annotations

import math
import struct
import zlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "extensions" / "signalrag-chromium" / "icons"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for size in (16, 32, 48, 128):
        (OUT / f"icon{size}.png").write_bytes(render_png(size))


def render_png(size: int) -> bytes:
    radius = size * 0.2
    rows = []
    for y in range(size):
        row = bytearray([0])
        for x in range(size):
            alpha = rounded_rect_alpha(x + 0.5, y + 0.5, size, radius)
            bg = mix((13, 128, 105), (37, 92, 153), y / max(size - 1, 1))
            if alpha < 255:
                row.extend((0, 0, 0, 0))
                continue
            color = bg
            if is_logo_stroke(x, y, size):
                color = (255, 255, 255)
            elif is_logo_crossbar(x, y, size):
                color = (183, 247, 229)
            row.extend((*color, 255))
        rows.append(bytes(row))

    raw = b"".join(rows)
    return b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0)),
            chunk(b"IDAT", zlib.compress(raw, 9)),
            chunk(b"IEND", b""),
        ]
    )


def rounded_rect_alpha(x: float, y: float, size: int, radius: float) -> int:
    inset = 0.0
    left = inset
    top = inset
    right = size - inset
    bottom = size - inset
    cx = min(max(x, left + radius), right - radius)
    cy = min(max(y, top + radius), bottom - radius)
    distance = math.hypot(x - cx, y - cy)
    return 255 if distance <= radius else 0


def is_logo_stroke(x: int, y: int, size: int) -> bool:
    width = max(2, round(size * 0.09))
    centers = (0.28, 0.5, 0.72)
    for cx in centers:
        if abs(x - size * cx) <= width / 2 and size * 0.24 <= y <= size * 0.76:
            return True
    return False


def is_logo_crossbar(x: int, y: int, size: int) -> bool:
    width = max(2, round(size * 0.08))
    if size * 0.23 <= x <= size * 0.77:
        return abs(y - size * 0.43) <= width / 2 or abs(y - size * 0.57) <= width / 2
    return False


def mix(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(round(a[i] * (1 - t) + b[i] * t) for i in range(3))


def chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


if __name__ == "__main__":
    main()
