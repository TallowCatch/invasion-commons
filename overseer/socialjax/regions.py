"""Regions of a SocialJax map for decentralised overseers (one overseer per region).

Layouts:
- "strips": R vertical strips of (nearly) equal width, split along the columns;
- "rows": R horizontal bands, split along the rows;
- "grid": R = a x b blocks (a rows of regions, b columns of regions, a <= b, as square as R allows), so R = 4
  gives quadrants.
Neighbours are regions that share an edge in the region layout (strips: left and right; grid: up, down, left, right).
"""
from __future__ import annotations

import numpy as np

LAYOUTS = ("strips", "rows", "grid")


def region_shape(regions: int, layout: str):
    """(region rows, region columns) of the layout."""
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}, not {layout!r}")
    if regions < 1:
        raise ValueError("need at least one region")
    if layout == "strips":
        return 1, regions
    if layout == "rows":
        return regions, 1
    a = int(np.floor(np.sqrt(regions)))
    while regions % a:
        a -= 1
    return a, regions // a


def region_grid(shape, regions: int, layout: str = "strips") -> np.ndarray:
    """(rows, cols) int array: the region index of every cell. Region ids run row-major over the layout."""
    rows, cols = shape
    a, b = region_shape(regions, layout)
    if a > rows or b > cols:
        raise ValueError(f"{regions} regions ({a} x {b}) do not fit a {rows} x {cols} map")
    r_idx = (np.arange(rows) * a) // rows
    c_idx = (np.arange(cols) * b) // cols
    return (r_idx[:, None] * b + c_idx[None, :]).astype(np.int32)


def neighbours(regions: int, layout: str = "strips") -> np.ndarray:
    """(R, R) bool: True where two regions share an edge."""
    a, b = region_shape(regions, layout)
    nb = np.zeros((regions, regions), bool)
    for r in range(regions):
        i, j = divmod(r, b)
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ii, jj = i + di, j + dj
            if 0 <= ii < a and 0 <= jj < b:
                nb[r, ii * b + jj] = True
    return nb
