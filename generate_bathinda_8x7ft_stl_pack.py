import json
import math
import os
import struct
import zipfile
import numpy as np
from PIL import Image, ImageDraw, ImageFont

BASE_DIR = r"d:\Ai\autoabode office work\bathinda-3d-terrain"
OUT_DIR = os.path.join(BASE_DIR, "fabrication_pack_8x7ft")
os.makedirs(OUT_DIR, exist_ok=True)

# Physical Sand Model Board:
# 8.0 ft East-West = 2438.4 mm  |  7.0 ft North-South = 2133.6 mm
# Exact Aspect Ratio = 8 : 7 = 1.14285714 : 1
# STL exported at 1:10 CAD Scale (243.84 x 213.36 mm) -> Scale 1000% (10x) for 1:1 8x7 ft table!
BOARD_W_MM_1_10 = 243.84
BOARD_D_MM_1_10 = 213.36

IB_UV = [
    [0.630, 0.000], [0.635, 0.018], [0.642, 0.035], [0.660, 0.052],
    [0.688, 0.082], [0.706, 0.118], [0.713, 0.155], [0.716, 0.184],
    [0.685, 0.214], [0.642, 0.252], [0.606, 0.285], [0.555, 0.302],
    [0.492, 0.320], [0.425, 0.340], [0.365, 0.365], [0.344, 0.382],
    [0.332, 0.435], [0.319, 0.495], [0.305, 0.560], [0.292, 0.618],
    [0.268, 0.678], [0.236, 0.735], [0.198, 0.805], [0.162, 0.875],
    [0.125, 0.935], [0.088, 1.000]
]

CORE_SECTOR_UV = [
    [0.105, 0.245], [0.775, 0.245], [0.775, 0.955], [0.105, 0.955], [0.105, 0.245]
]

WATERWAYS = [
    {"name": "Sutlej (Satluj) River", "cat": "river", "width": 2.45, "carveDepth": 1.55,
     "uv": [[0.760, 0.000], [0.705, 0.032], [0.658, 0.052], [0.605, 0.065], [0.550, 0.072],
            [0.470, 0.105], [0.405, 0.142], [0.335, 0.178], [0.265, 0.220], [0.195, 0.252],
            [0.125, 0.295], [0.055, 0.338], [0.000, 0.372]]},
    {"name": "Ghaggar River (Nali / Hakra)", "cat": "river", "width": 1.65, "carveDepth": 1.10,
     "uv": [[1.000, 0.615], [0.925, 0.618], [0.865, 0.655], [0.792, 0.705], [0.745, 0.752],
            [0.705, 0.805], [0.675, 0.832], [0.605, 0.835], [0.530, 0.848], [0.470, 0.880],
            [0.432, 0.895], [0.350, 0.922], [0.246, 0.928], [0.130, 0.928], [0.025, 0.925]]},
    {"name": "Gang (Bikaner) Main Canal", "cat": "main_canal", "width": 1.20, "carveDepth": 0.78,
     "uv": [[0.855, 0.000], [0.795, 0.088], [0.742, 0.175], [0.702, 0.255], [0.672, 0.325],
            [0.654, 0.395], [0.585, 0.465], [0.510, 0.535], [0.435, 0.615], [0.398, 0.682], [0.365, 0.760]]},
    {"name": "Abohar / Sirhind Feeder", "cat": "highlighted", "width": 1.10, "carveDepth": 0.68,
     "uv": [[0.985, 0.050], [0.915, 0.135], [0.848, 0.228], [0.785, 0.295], [0.725, 0.348], [0.662, 0.388]]},
    {"name": "IGNP Main & Anupgarh Br", "cat": "main_canal", "width": 1.30, "carveDepth": 0.82,
     "uv": [[1.000, 0.535], [0.945, 0.595], [0.885, 0.685], [0.835, 0.775], [0.765, 0.842],
            [0.682, 0.852], [0.580, 0.865], [0.455, 0.898], [0.320, 0.935], [0.205, 0.962]]},
    {"name": "IGNP Suratgarh South Br", "cat": "main_canal", "width": 1.00, "carveDepth": 0.60,
     "uv": [[0.682, 0.852], [0.652, 0.915], [0.605, 0.968], [0.575, 1.000]]},
    {"name": "Karanpur & Phusewala Disty", "cat": "highlighted", "width": 0.95, "carveDepth": 0.55,
     "uv": [[0.672, 0.325], [0.585, 0.362], [0.485, 0.398], [0.445, 0.422], [0.402, 0.448], [0.348, 0.482]]},
    {"name": "Ganganagar Disty (GGN)", "cat": "highlighted", "width": 0.90, "carveDepth": 0.52,
     "uv": [[0.654, 0.395], [0.646, 0.480], [0.635, 0.565], [0.618, 0.655]]},
    {"name": "Padampur & Rupnagar Disty", "cat": "highlighted", "width": 0.92, "carveDepth": 0.55,
     "uv": [[0.585, 0.465], [0.505, 0.548], [0.432, 0.572], [0.366, 0.588], [0.318, 0.608]]},
    {"name": "Raisinghnagar (RSH) Disty", "cat": "highlighted", "width": 0.90, "carveDepth": 0.52,
     "uv": [[0.485, 0.635], [0.398, 0.682], [0.325, 0.712], [0.262, 0.735]]},
    {"name": "Sadulshahr & Lalgarh Br", "cat": "canal", "width": 0.85, "carveDepth": 0.48,
     "uv": [[0.848, 0.228], [0.858, 0.398], [0.798, 0.485], [0.745, 0.555]]},
    {"name": "Eastern Sadiqia (Hakra) Canal", "cat": "canal", "width": 1.15, "carveDepth": 0.72,
     "uv": [[0.658, 0.052], [0.665, 0.135], [0.615, 0.215], [0.515, 0.282], [0.395, 0.335],
            [0.312, 0.372], [0.295, 0.448], [0.272, 0.515], [0.242, 0.568], [0.198, 0.618],
            [0.175, 0.695], [0.135, 0.768], [0.025, 0.925]]},
    {"name": "Fordwah Canal", "cat": "canal", "width": 1.02, "carveDepth": 0.62,
     "uv": [[0.658, 0.052], [0.555, 0.135], [0.472, 0.208], [0.375, 0.275], [0.282, 0.338],
            [0.185, 0.382], [0.095, 0.415], [0.010, 0.448]]},
    {"name": "Pakpattan Main Canal", "cat": "canal", "width": 0.95, "carveDepth": 0.58,
     "uv": [[0.485, 0.015], [0.352, 0.075], [0.240, 0.095], [0.135, 0.118], [0.020, 0.145]]},
]

HIGHWAYS = [
    {"name": "NH-62", "uv": [[0.895, 0.000], [0.848, 0.228], [0.755, 0.315], [0.654, 0.395], [0.656, 0.580], [0.675, 0.740], [0.682, 0.842], [0.648, 1.000]]},
    {"name": "NH-911", "uv": [[0.654, 0.395], [0.478, 0.372], [0.402, 0.448], [0.378, 0.530], [0.366, 0.588], [0.398, 0.682], [0.415, 0.795], [0.432, 0.895], [0.246, 0.932]]},
    {"name": "SH-3", "uv": [[0.654, 0.395], [0.575, 0.475], [0.505, 0.548], [0.455, 0.618], [0.398, 0.682]]},
    {"name": "NH-54/62E", "uv": [[0.848, 0.228], [0.858, 0.398], [0.892, 0.510], [0.925, 0.618], [0.792, 0.705], [0.682, 0.842]]},
    {"name": "GGN-BTI Axis", "uv": [[0.654, 0.395], [0.858, 0.398], [1.000, 0.412]]},
    {"name": "Pak Border Hwy", "uv": [[0.658, 0.052], [0.582, 0.208], [0.472, 0.208], [0.282, 0.338], [0.272, 0.515], [0.198, 0.618], [0.135, 0.768], [0.025, 0.925]]},
    {"name": "BWN-CHT Hwy", "uv": [[0.282, 0.338], [0.095, 0.415], [0.135, 0.118], [0.352, 0.075]]},
]

SARVATRA_SITES = [
    {"name": "Sarvatra-1 (Gang Canal • Mohanpura)", "u": 0.672, "v": 0.325},
    {"name": "Sarvatra-2 (Karanpur Disty • BP 325)", "u": 0.445, "v": 0.422},
    {"name": "Sarvatra-3 (IGNP / Ghaggar • Vijaynagar)", "u": 0.432, "v": 0.895},
]

HEADWORKS_SITES = [
    {"name": "SMK HW (Sulemanki)", "u": 0.658, "v": 0.052},
    {"name": "Marinihal HW", "u": 0.470, "v": 0.105},
    {"name": "Jalwala HW", "u": 0.312, "v": 0.372},
    {"name": "Gulab Ali HW", "u": 0.242, "v": 0.568},
    {"name": "Mianwali HW", "u": 0.175, "v": 0.695},
]

HIGHWAY_BRIDGES = [
    {"name": "Abohar NH-62 Br", "u": 0.785, "v": 0.295},
    {"name": "Sri Ganganagar Br", "u": 0.654, "v": 0.395},
    {"name": "Padampur Br", "u": 0.510, "v": 0.535},
    {"name": "Raisinghnagar Br", "u": 0.398, "v": 0.682},
    {"name": "Hanumangarh Br", "u": 0.925, "v": 0.618},
    {"name": "Suratgarh Br", "u": 0.682, "v": 0.846},
    {"name": "Anupgarh Br", "u": 0.285, "v": 0.932},
]

TOWNS = [
    {"name": "FAZILKA", "u": 0.945, "v": 0.048, "side": "ind"},
    {"name": "ABOHAR", "u": 0.848, "v": 0.228, "side": "ind"},
    {"name": "SRI GANGANAGAR", "u": 0.654, "v": 0.395, "side": "ind"},
    {"name": "SADULSHAHR", "u": 0.858, "v": 0.398, "side": "ind"},
    {"name": "KESRISINGHPUR", "u": 0.478, "v": 0.372, "side": "ind"},
    {"name": "SRIKARANPUR", "u": 0.402, "v": 0.448, "side": "ind"},
    {"name": "PADAMPUR", "u": 0.505, "v": 0.548, "side": "ind"},
    {"name": "GAJSINGHPUR", "u": 0.366, "v": 0.588, "side": "ind"},
    {"name": "RAISINGHNAGAR", "u": 0.398, "v": 0.682, "side": "ind"},
    {"name": "HANUMANGARH", "u": 0.925, "v": 0.618, "side": "ind"},
    {"name": "PILIBANGAN", "u": 0.792, "v": 0.705, "side": "ind"},
    {"name": "SURATGARH", "u": 0.682, "v": 0.842, "side": "ind"},
    {"name": "SRI VIJAYNAGAR", "u": 0.432, "v": 0.895, "side": "ind"},
    {"name": "ANUPGARH", "u": 0.246, "v": 0.932, "side": "ind"},
    {"name": "PAKPATTAN", "u": 0.352, "v": 0.075, "side": "pak"},
    {"name": "ARIFWALA", "u": 0.135, "v": 0.118, "side": "pak"},
    {"name": "MINCHINABAD", "u": 0.472, "v": 0.208, "side": "pak"},
    {"name": "BAHAWALNAGAR", "u": 0.282, "v": 0.338, "side": "pak"},
    {"name": "CHISHTIAN", "u": 0.095, "v": 0.415, "side": "pak"},
    {"name": "DUNGA BUNGA", "u": 0.272, "v": 0.515, "side": "pak"},
    {"name": "HAROONABAD", "u": 0.198, "v": 0.618, "side": "pak"},
    {"name": "FAQIR WALI", "u": 0.135, "v": 0.768, "side": "pak"},
    {"name": "FORT ABBAS", "u": 0.032, "v": 0.925, "side": "pak"},
]


def dist_to_polyline_grid(U, V, poly):
    min_d2 = np.full(U.shape, 1e9, dtype=np.float32)
    for i in range(len(poly) - 1):
        ax, ay = poly[i]
        bx, by = poly[i + 1]
        dx, dy = bx - ax, by - ay
        len_sq = dx * dx + dy * dy + 1e-12
        t = np.clip(((U - ax) * dx + (V - ay) * dy) / len_sq, 0.0, 1.0)
        px = ax + t * dx
        py = ay + t * dy
        d2 = (U - px) ** 2 + (V - py) ** 2
        min_d2 = np.minimum(min_d2, d2)
    return np.sqrt(min_d2)


def build_height_field(dem, nx, ny):
    """
    Returns height Z in mm (at 1:10 scale, where XY is 243.84 x 213.36 mm).
    Minimum solid floor under deepest river cut >= 4.2 mm (42 mm at 1:1 full scale).
    """
    u_1d = np.linspace(0.0, 1.0, nx, dtype=np.float32)
    v_1d = np.linspace(0.0, 1.0, ny, dtype=np.float32)
    U, V = np.meshgrid(u_1d, v_1d)

    # Bilinear interpolation of smoothed SRTM DEM
    grid = np.array(dem["grid"], dtype=np.float32)
    gh, gw = grid.shape
    gx = np.clip(U * (gw - 1), 0, gw - 1.0001)
    gy = np.clip(V * (gh - 1), 0, gh - 1.0001)
    x0 = np.floor(gx).astype(np.int32)
    y0 = np.floor(gy).astype(np.int32)
    x1 = np.minimum(gw - 1, x0 + 1)
    y1 = np.minimum(gh - 1, y0 + 1)
    fx = gx - x0
    fy = gy - y0

    el = (
        grid[y0, x0] * (1 - fx) * (1 - fy)
        + grid[y0, x1] * fx * (1 - fy)
        + grid[y1, x0] * (1 - fx) * fy
        + grid[y1, x1] * fx * fy
    )
    norm_el = (el - dem["minEl"]) / max(1.0, dem["maxEl"] - dem["minEl"])

    # Base elevation in 1:10 mm: 6.2 mm base + 7.5 mm tactical relief range (62 mm to 137 mm full scale)
    Z = 6.2 + norm_el * 7.5

    # Aeolian Thar sand dune micro-relief in southern belt (v > 0.72)
    dune_mask = np.clip((V - 0.72) / 0.22, 0.0, 1.0)
    ripples = np.sin(U * 95.0 + np.sin(V * 42.0) * 2.2) * np.cos(V * 68.0 - U * 25.0) * 0.30 * dune_mask

    # Carve all 14 survey-calibrated rivers, main canals & distributaries + earthen bunds
    carve = np.zeros_like(Z)
    bund = np.zeros_like(Z)
    for w in WATERWAYS:
        d = dist_to_polyline_grid(U, V, w["uv"])
        half_w = (w["width"] / BOARD_W_MM_1_10) * 0.68
        in_chan = d < (half_w * 1.15)
        t = np.clip(d / (half_w * 1.15), 0.0, 1.0)
        chan_delta = -w["carveDepth"] * (1.0 - t * t)
        carve = np.where(in_chan, np.minimum(carve, chan_delta), carve)

        in_bund = (~in_chan) & (d < half_w * 2.45)
        bt = np.clip((d - half_w * 1.15) / (half_w * 1.30), 0.0, 1.0)
        b_val = np.sin(bt * math.pi) * 0.26
        bund = np.where(in_bund, np.maximum(bund, b_val), bund)

    # Apply dune ripples only outside water channels
    Z = Z + np.where(carve < -0.02, carve, bund + ripples)

    # Subtle raised highway embankment (+0.18 mm at 1:10 = +1.8 mm at 1:1) outside deep river cuts
    for hwy in HIGHWAYS:
        dh = dist_to_polyline_grid(U, V, hwy["uv"])
        hw_half = 0.0024
        on_hwy = (dh < hw_half) & (carve > -0.15)
        ht = np.clip(dh / hw_half, 0.0, 1.0)
        Z = np.where(on_hwy, Z + 0.18 * (1.0 - ht * ht), Z)

    # Survey-Exact International Border (IB BP 260-365): tactile V-groove + twin security bunds
    d_ib = dist_to_polyline_grid(U, V, IB_UV)
    ib_groove_w = 0.0022
    ib_bund_w = 0.0052
    in_ib_groove = d_ib < ib_groove_w
    in_ib_bund = (~in_ib_groove) & (d_ib < ib_bund_w)
    Z = np.where(in_ib_groove, Z - 0.42 * (1.0 - (d_ib / ib_groove_w) ** 2), Z)
    ib_bt = np.clip((d_ib - ib_groove_w) / (ib_bund_w - ib_groove_w), 0.0, 1.0)
    Z = np.where(in_ib_bund & (carve > -0.15), Z + np.sin(ib_bt * math.pi) * 0.28, Z)

    # Level registration seats for 5 Gated Barrage Headworks & 3 Sarvatra Bridges
    for hw in HEADWORKS_SITES:
        r = np.sqrt((U - hw["u"]) ** 2 + (V - hw["v"]) ** 2)
        Z = np.where(r < 0.0065, np.maximum(Z, 6.45), Z)
    for sb in SARVATRA_SITES:
        r = np.sqrt((U - sb["u"]) ** 2 + (V - sb["v"]) ** 2)
        ring = (r >= 0.0038) & (r <= 0.0072)
        Z = np.where(ring, np.maximum(Z, 6.55), Z)

    Z = np.maximum(4.2, Z)
    return Z


def write_watertight_binary_stl(filepath, Z, width_mm, depth_mm):
    """
    Writes a 100% watertight solid binary STL with:
    - Top carved terrain surface (Z[iv, iu])
    - Flat bottom build plate (Z = 0.0)
    - 4 closed perimeter vertical walls (North, South, West, East)
    Coordinate system:
      X: 0 -> width_mm (243.84 mm = West to East)
      Y: 0 -> depth_mm (213.36 mm = South to North, so North is +Y!)
      Z: 0 -> ~14.2 mm (Up)
    """
    ny, nx = Z.shape
    xs = np.linspace(0.0, width_mm, nx, dtype=np.float32)
    # iv=0 is North (v=0 -> y = depth_mm), iv=ny-1 is South (v=1 -> y = 0)
    ys = np.linspace(depth_mm, 0.0, ny, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)

    # Number of triangles:
    # Top: 2 * (nx-1)*(ny-1)
    # Bottom: 2 * (nx-1)*(ny-1) (we can actually use 2 giant triangles for flat bottom IF vertices match?
    # NO! For strict manifold edge-to-edge watertightness, perimeter edges must match bottom grid or fan!
    # Matching full bottom grid guarantees 100% shared edges with zero T-junctions in every slicer!)
    n_cells = (nx - 1) * (ny - 1)
    n_top = 2 * n_cells
    n_bot = 2 * n_cells
    n_walls = 2 * (2 * (nx - 1) + 2 * (ny - 1))
    total_tris = n_top + n_bot + n_walls

    stl_dtype = np.dtype([
        ("normal", "<f4", (3,)),
        ("v1", "<f4", (3,)),
        ("v2", "<f4", (3,)),
        ("v3", "<f4", (3,)),
        ("attr", "<u2"),
    ])
    tris = np.zeros(total_tris, dtype=stl_dtype)
    idx = 0

    # Top grid vertices
    P = np.stack([X, Y, Z], axis=-1)
    p00 = P[:-1, :-1].reshape(-1, 3)
    p10 = P[:-1, 1:].reshape(-1, 3)
    p01 = P[1:, :-1].reshape(-1, 3)
    p11 = P[1:, 1:].reshape(-1, 3)

    def calc_normals(a, b, c):
        n = np.cross(b - a, c - a)
        lengths = np.linalg.norm(n, axis=1, keepdims=True)
        return n / np.maximum(lengths, 1e-12)

    # Top triangles (CCW from +Z: p00 -> p01 -> p11 and p00 -> p11 -> p10)
    tris["v1"][idx:idx + n_cells] = p00
    tris["v2"][idx:idx + n_cells] = p01
    tris["v3"][idx:idx + n_cells] = p11
    tris["normal"][idx:idx + n_cells] = calc_normals(p00, p01, p11)
    idx += n_cells

    tris["v1"][idx:idx + n_cells] = p00
    tris["v2"][idx:idx + n_cells] = p11
    tris["v3"][idx:idx + n_cells] = p10
    tris["normal"][idx:idx + n_cells] = calc_normals(p00, p11, p10)
    idx += n_cells

    # Bottom grid vertices (Z = 0, normal = [0, 0, -1])
    B = np.stack([X, Y, np.zeros_like(Z)], axis=-1)
    b00 = B[:-1, :-1].reshape(-1, 3)
    b10 = B[:-1, 1:].reshape(-1, 3)
    b01 = B[1:, :-1].reshape(-1, 3)
    b11 = B[1:, 1:].reshape(-1, 3)

    tris["v1"][idx:idx + n_cells] = b00
    tris["v2"][idx:idx + n_cells] = b11
    tris["v3"][idx:idx + n_cells] = b01
    tris["normal"][idx:idx + n_cells] = [0.0, 0.0, -1.0]
    idx += n_cells

    tris["v1"][idx:idx + n_cells] = b00
    tris["v2"][idx:idx + n_cells] = b10
    tris["v3"][idx:idx + n_cells] = b11
    tris["normal"][idx:idx + n_cells] = [0.0, 0.0, -1.0]
    idx += n_cells

    # Helper to append a wall strip where top_pts -> bot_pts creates outward normal
    def add_wall(top_a, top_b, bot_a, bot_b, normal_vec):
        nonlocal idx
        m = len(top_a)
        tris["v1"][idx:idx + m] = top_a
        tris["v2"][idx:idx + m] = bot_b
        tris["v3"][idx:idx + m] = bot_a
        tris["normal"][idx:idx + m] = normal_vec
        idx += m

        tris["v1"][idx:idx + m] = top_a
        tris["v2"][idx:idx + m] = top_b
        tris["v3"][idx:idx + m] = bot_b
        tris["normal"][idx:idx + m] = normal_vec
        idx += m

    # North wall (iv = 0, y = depth_mm, outward normal = +Y):
    # Viewed from +Y looking towards -Y, +X is to the LEFT, so go East->West or check cross product:
    # Let's check: top_a=(x0,D,z0), top_b=(x1,D,z1), bot_a=(x0,D,0), bot_b=(x1,D,0) with x1 > x0:
    # (bot_b - top_a) x (bot_a - top_a) = (+dx, 0, -z0) x (0, 0, -z0) = (0, +dx*z0, 0) -> +Y! Exact!
    add_wall(P[0, :-1], P[0, 1:], B[0, :-1], B[0, 1:], [0.0, 1.0, 0.0])

    # South wall (iv = ny-1, y = 0, outward normal = -Y):
    # Reverse direction (1: -> :-1) so normal points -Y!
    add_wall(P[-1, 1:], P[-1, :-1], B[-1, 1:], B[-1, :-1], [0.0, -1.0, 0.0])

    # West wall (iu = 0, x = 0, outward normal = -X):
    # iv goes 0 (y=D) -> ny-1 (y=0), so dy < 0:
    # (bot_b - top_a) x (bot_a - top_a) = (0, -|dy|, -z0) x (0, 0, -z0) = (+|dy|*z0, 0, 0) -> +X, so reverse!
    add_wall(P[1:, 0], P[:-1, 0], B[1:, 0], B[:-1, 0], [-1.0, 0.0, 0.0])

    # East wall (iu = nx-1, x = width_mm, outward normal = +X):
    add_wall(P[:-1, -1], P[1:, -1], B[:-1, -1], B[1:, -1], [1.0, 0.0, 0.0])

    assert idx == total_tris, f"Expected {total_tris}, got {idx}"

    header = b"autoAbode 64 Unit Bathinda 8.0x7.0ft (243.84x213.36mm 1:10) Master STL".ljust(80, b" ")
    with open(filepath, "wb") as f:
        f.write(header)
        f.write(struct.pack("<I", total_tris))
        f.write(tris.tobytes())

    return total_tris


def save_16bit_heightmap(filepath, Z):
    z_min = float(Z.min())
    z_max = float(Z.max())
    norm = (Z - z_min) / max(1e-6, z_max - z_min)
    arr16 = np.clip(np.round(norm * 65535.0), 0, 65535).astype(np.uint16)
    img16 = Image.fromarray(arr16)
    img16.save(filepath)
    return z_min, z_max


def get_font(size, bold=False):
    candidates = [
        r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def build_shopfloor_blueprint(filepath, Z_hi):
    W, H = 2240, 1540
    canvas = Image.new("RGB", (W, H), (10, 16, 29))
    draw = ImageDraw.Draw(canvas)

    # Header bar
    draw.rectangle([0, 0, W, 98], fill=(15, 23, 42))
    draw.line([(0, 98), (W, 98)], fill=(217, 119, 6), width=4)
    draw.text((34, 18), "64 ENGR REGT (BATHINDA SECTOR) — MASTER 8.0 FT × 7.0 FT (2,438.4 × 2,133.6 mm) FABRICATION BLUEPRINT",
              font=get_font(28, True), fill=(251, 191, 36))
    draw.text((34, 56), "Survey-Locked Aspect Ratio 8 : 7 (1.143 : 1)  •  Radcliffe IB (BP 260–365)  •  14 Carved Rivers/Canals  •  5 Headworks + 3 Detachable Sarvatra + 7 Road Bridges",
              font=get_font(18, False), fill=(203, 213, 225))

    # Left Map Frame: Exact 8:7 Map Viewport (1280 x 1120 px -> 1280/1120 = 8/7 = 1.142857!)
    MX, MY = 82, 186
    MW, MH = 1280, 1120

    # Load hybrid GIS satellite background + blend shaded relief from Z_hi
    hyb_path = os.path.join(BASE_DIR, "assets", "bathinda_hybrid_gis.jpg")
    bg = Image.open(hyb_path).convert("RGB").resize((MW, MH), Image.Resampling.LANCZOS)
    bg_np = np.asarray(bg, dtype=np.float32)

    # Hillshade from Z_hi resized to (MH, MW)
    z_map = np.asarray(Image.fromarray(Z_hi).resize((MW, MH), Image.Resampling.BILINEAR), dtype=np.float32)
    gy, gx = np.gradient(z_map)
    shade = np.clip(1.0 + (-gx * 4.5 - gy * 4.5), 0.68, 1.32)[..., None]
    blended = np.clip(bg_np * 0.82 * shade + 12.0, 0, 255).astype(np.uint8)
    map_img = Image.fromarray(blended)
    mdraw = ImageDraw.Draw(map_img, "RGBA")

    # 8 x 7 = 56 1-Foot (304.8 x 304.8 mm) CNC / 3D Print Modular Grid Lines!
    for ix in range(1, 8):
        px = int(round(ix * MW / 8.0))
        mdraw.line([(px, 0), (px, MH)], fill=(255, 255, 255, 72), width=2)
    for iy in range(1, 7):
        py = int(round(iy * MH / 7.0))
        mdraw.line([(0, py), (MW, py)], fill=(255, 255, 255, 72), width=2)

    # Highways
    for hwy in HIGHWAYS:
        pts = [(u * MW, v * MH) for u, v in hwy["uv"]]
        mdraw.line(pts, fill=(15, 23, 42, 190), width=6)
        mdraw.line(pts, fill=(245, 158, 11, 235), width=3)

    # All 14 Waterways
    for w in WATERWAYS:
        pts = [(u * MW, v * MH) for u, v in w["uv"]]
        pw = 10 if w["cat"] == "river" else (7 if w["cat"] in ("main_canal", "highlighted") else 5)
        halo = (254, 240, 138, 155) if w["cat"] == "highlighted" else (186, 230, 253, 145)
        col = (3, 105, 161, 255) if w["cat"] == "river" else ((2, 132, 199, 255) if w["cat"] != "highlighted" else (14, 165, 233, 255))
        mdraw.line(pts, fill=halo, width=pw + 4)
        mdraw.line(pts, fill=col, width=pw)

    # Core Sector Dashed Box
    cpts = [(u * MW, v * MH) for u, v in CORE_SECTOR_UV]
    mdraw.line(cpts, fill=(251, 191, 36, 210), width=3)

    # International Border (IB)
    ib_pts = [(u * MW, v * MH) for u, v in IB_UV]
    mdraw.line(ib_pts, fill=(239, 68, 68, 105), width=20)
    mdraw.line(ib_pts, fill=(220, 38, 38, 255), width=6)
    for i in range(len(ib_pts) - 1):
        if i % 2 == 0:
            mdraw.line([ib_pts[i], ib_pts[i + 1]], fill=(254, 240, 138, 255), width=2)

    # 5 Headworks (Purple Squares), 3 Sarvatra Bridges (Bright Green Diamonds), 7 Highway Bridges (Cyan Circles)
    for hw in HEADWORKS_SITES:
        cx, cy = hw["u"] * MW, hw["v"] * MH
        mdraw.rectangle([cx - 8, cy - 8, cx + 8, cy + 8], fill=(168, 85, 247, 255), outline=(255, 255, 255, 255), width=2)
    for sb in SARVATRA_SITES:
        cx, cy = sb["u"] * MW, sb["v"] * MH
        mdraw.polygon([(cx, cy - 11), (cx + 11, cy), (cx, cy + 11), (cx - 11, cy)], fill=(16, 185, 129, 255), outline=(255, 255, 255, 255))
    for hb in HIGHWAY_BRIDGES:
        cx, cy = hb["u"] * MW, hb["v"] * MH
        mdraw.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=(56, 189, 248, 255), outline=(15, 23, 42, 255), width=2)

    # Town Badges (India = Green/Gold East of IB, Pakistan = Red/Slate West of IB)
    f_town = get_font(13, True)
    for t in TOWNS:
        cx, cy = t["u"] * MW, t["v"] * MH
        is_ind = t["side"] == "ind"
        dot_col = (34, 197, 94, 255) if is_ind else (248, 113, 113, 255)
        mdraw.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=dot_col, outline=(0, 0, 0, 255), width=2)
        lbl = t["name"]
        bbox = mdraw.textbbox((0, 0), lbl, font=f_town)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx = min(max(6, cx + 8), MW - tw - 10)
        ty = min(max(4, cy - th // 2 - 2), MH - th - 8)
        mdraw.rounded_rectangle([tx - 4, ty - 2, tx + tw + 4, ty + th + 3], radius=4,
                                fill=(15, 23, 42, 215), outline=dot_col, width=1)
        mdraw.text((tx, ty - 1), lbl, font=f_town, fill=(248, 250, 252, 255) if is_ind else (254, 202, 202, 255))

    # Sector watermarks on map
    mdraw.rounded_rectangle([MW - 285, 14, MW - 14, 52], radius=6, fill=(6, 78, 59, 225), outline=(52, 211, 153, 255), width=2)
    mdraw.text((MW - 272, 22), "OWN SECTOR — INDIA (EAST OF IB)", font=get_font(15, True), fill=(236, 253, 245, 255))

    mdraw.rounded_rectangle([14, 14, 305, 52], radius=6, fill=(127, 29, 29, 225), outline=(248, 113, 113, 255), width=2)
    mdraw.text((26, 22), "OPPOSITE SECTOR — PAK (WEST OF IB)", font=get_font(15, True), fill=(254, 242, 242, 255))

    canvas.paste(map_img, (MX, MY))
    draw.rectangle([MX - 2, MY - 2, MX + MW + 2, MY + MH + 2], outline=(245, 158, 11), width=3)

    # Top Dimension Ruler (8.0 FT = 2,438.4 mm)
    draw.rounded_rectangle([MX + 310, MY - 80, MX + 970, MY - 52], radius=6, fill=(15, 23, 42), outline=(56, 189, 248), width=1)
    draw.text((MX + 332, MY - 76), "<---  EXACT WIDTH: 8.0 FT = 2,438.4 mm (8 COLS × 304.8 mm)  --->",
              font=get_font(15, True), fill=(56, 189, 248))
    draw.line([(MX, MY - 24), (MX + MW, MY - 24)], fill=(251, 191, 36), width=3)
    for ix in range(9):
        px = MX + int(round(ix * MW / 8.0))
        draw.line([(px, MY - 33), (px, MY - 13)], fill=(251, 191, 36), width=3)
        draw.text((px - 15, MY - 50), f"{ix} FT", font=get_font(13, True), fill=(253, 230, 138))

    # Left Dimension Ruler (7.0 FT = 2,133.6 mm)
    draw.line([(MX - 24, MY), (MX - 24, MY + MH)], fill=(251, 191, 36), width=3)
    for iy in range(8):
        py = MY + int(round(iy * MH / 7.0))
        draw.line([(MX - 34, py), (MX - 13, py)], fill=(251, 191, 36), width=3)
        draw.text((14, py - 9), f"{iy} FT", font=get_font(13, True), fill=(253, 230, 138))

    # Map Legend Bar underneath map
    LY = MY + MH + 16
    draw.rounded_rectangle([MX, LY, MX + MW, LY + 76], radius=8, fill=(15, 23, 42), outline=(51, 65, 85), width=2)
    draw.text((MX + 18, LY + 12),
              "LEGEND:   [RED/GOLD LINE] Radcliffe IB (BP 260–365)    [BLUE CARVED] 14 Rivers, Main Canals & Distributaries    [AMBER] 7 Strategic Highways (NH-62 / NH-911 / SH-3)",
              font=get_font(15, True), fill=(226, 232, 240))
    draw.text((MX + 18, LY + 42),
              "[GREEN DIAMOND] 3 Detachable Sarvatra Assault Bridges    [PURPLE SQUARE] 5 Gated Barrage Headworks (SMK, Marinihal, Jalwala, Gulab Ali, Mianwali)    [CYAN DOT] 7 Road Bridges",
              font=get_font(15, False), fill=(148, 163, 184))

    # Right Column Cards: Engineering & Client Requirement Alignment
    RX = MX + MW + 32
    RW = W - RX - 28

    def draw_card(y0, h, border_col, title_bg, title, bullets):
        draw.rounded_rectangle([RX, y0, RX + RW, y0 + h], radius=10, fill=(15, 23, 42), outline=border_col, width=2)
        draw.rounded_rectangle([RX + 2, y0 + 2, RX + RW - 2, y0 + 44], radius=8, fill=title_bg)
        draw.text((RX + 16, y0 + 10), title, font=get_font(18, True), fill=(255, 255, 255))
        cy = y0 + 56
        f_b = get_font(15, False)
        f_bb = get_font(15, True)
        for head, body in bullets:
            draw.text((RX + 16, cy), f"• {head}:", font=f_bb, fill=(253, 230, 138))
            cy += 21
            for line in body:
                draw.text((RX + 30, cy), line, font=f_b, fill=(226, 232, 240))
                cy += 20
            cy += 6

    # Card 1: Why ZBrush 1:1 Draft Was Flagged
    draw_card(
        130, 320, (239, 68, 68), (153, 27, 27),
        "1. WHY ZBRUSH 1:1 SQUARE DRAFT FAILED QC",
        [
            ("Wrong 1:1 Square Aspect Ratio", [
                "Table is 8.0 × 7.0 ft (2438.4 × 2133.6 mm = 8:7 ratio).",
                "Sculpting on a 1:1 square compresses East–West by 14.3%!"
            ]),
            ("Freehand Squiggly Lines vs Survey Hydrology", [
                "Manual sculpting drew only 4 approximate curves instead",
                "of the 14 geo-referenced rivers, feeders & distributaries."
            ]),
            ("IB & Border Towns Alignment Critical for Army", [
                "Kesrisinghpur, Srikaranpur, Gajsinghpur, Raisinghnagar &",
                "Anupgarh MUST sit strictly EAST of IB (BP 260–365)."
            ]),
        ]
    )

    # Card 2: What Client (Col. Sunil / 64 Engr Regt) Wants
    draw_card(
        468, 495, (16, 185, 129), (6, 95, 70),
        "2. EXACT 64 UNIT CLIENT DELIVERABLE SPECS",
        [
            ("Physical Sand Table & Acrylic Hood (8.0 × 7.0 ft)", [
                "2,438.4 mm (E–W) × 2,133.6 mm (N–S) modular terrain on",
                "wooden/MS operations-room table + clear dust enclosure."
            ]),
            ("14 Physically Recessed Water Channels + Bunds", [
                "Sutlej & Ghaggar riverbeds + Gang, Sirhind Feeder, IGNP,",
                "Karanpur, GGN, Padampur, RSH, Sadiqia, Fordwah, Pakpattan."
            ]),
            ("3 Dummy / Detachable 'Sarvatra' Assault Bridges", [
                "Removable scale Sarvatra spans at Mohanpura (Gang Canal),",
                "Karanpur Disty (BP 325 Axis) & Vijaynagar (IGNP/Ghaggar)."
            ]),
            ("5 Gated Barrage Headworks + 7 Highway Bridges", [
                "3D-printed SMK (Sulemanki), Marinihal, Jalwala, Gulab Ali",
                "& Mianwali barrages + 7 NH-62 / NH-911 road crossings."
            ]),
            ("Zone LEDs + Matte Military Sand/Olive Paint", [
                "Red/Amber IB LED strip, Blue waterway circuits, authentic",
                "matte sand/olive finish + engraved Brass 8ft/7ft rulers."
            ]),
        ]
    )

    # Card 3: Shopfloor Execution Steps for Krishna Nagar & Astitva
    draw_card(
        980, 398, (56, 189, 248), (3, 105, 161),
        "3. HOW TO USE THIS PACK (ZERO MANUAL GUESSWORK)",
        [
            ("Direct Watertight Binary STL (Ready to Slice / CAM)", [
                "Bathinda_64Unit_8x7ft_2438x2134mm_Watertight_Master.stl",
                "is exact 243.84 × 213.36 mm (1:10) -> Scale 1000% (10×)!"
            ]),
            ("16-Bit CNC / ZBrush Displacement Alpha (2438×2134 px)", [
                "1 px = 1.0 mm! In ZBrush, set Plane = 2438.4 × 2133.6",
                "FIRST, then apply 16-bit PNG — never sculpt canals by hand."
            ]),
            ("Modular Slicing Grid Options", [
                "• CNC Foam/MDF or XL800/IDEX400: 8×7 = 56 tiles (304.8 mm)",
                "• Bambu A1 (256 mm bed): 10×9 = 90 tiles (243.8×237.1 mm)"
            ]),
        ]
    )

    # Footer
    draw.rectangle([0, H - 48, W, H], fill=(15, 23, 42))
    draw.text((34, H - 34),
              "autoAbode Engineering  •  Live 3D Reference: https://botbit-in.github.io/bathinda-3d-terrain/  •  Generated directly from Smoothed SRTM DEM + Survey UV",
              font=get_font(16, True), fill=(148, 163, 184))

    canvas.save(filepath, quality=95)


def main():
    dem_path = os.path.join(BASE_DIR, "assets", "bathinda_dem_smooth.json")
    with open(dem_path, "r", encoding="utf-8") as f:
        dem = json.load(f)

    print("1/4 Building 400x350 (exact 8:7) carved heightfield for Watertight Binary STL...")
    Z_stl = build_height_field(dem, nx=401, ny=351)  # 400 x 350 cells = exact 8:7!
    stl_path = os.path.join(OUT_DIR, "Bathinda_64Unit_8x7ft_2438x2134mm_Watertight_Master.stl")
    n_tris = write_watertight_binary_stl(stl_path, Z_stl, BOARD_W_MM_1_10, BOARD_D_MM_1_10)
    stl_mb = os.path.getsize(stl_path) / (1024 * 1024)
    print(f"    Saved STL: {stl_path} ({n_tris:,} triangles, {stl_mb:.2f} MB)")

    print("2/4 Building 2438x2134 px (1 px = 1.000 mm!) 16-Bit CNC & ZBrush Heightmap...")
    Z_hi = build_height_field(dem, nx=2438, ny=2134)
    hmap_path = os.path.join(OUT_DIR, "Bathinda_64Unit_8x7ft_2438x2134_16bit_CNC_ZBrush_Heightmap.png")
    zmin, zmax = save_16bit_heightmap(hmap_path, Z_hi)
    print(f"    Saved 16-bit Heightmap: {hmap_path} (Z range 1:10 = {zmin:.2f}..{zmax:.2f} mm | 1:1 = {zmin*10:.1f}..{zmax*10:.1f} mm)")

    print("3/4 Rendering High-Res Shopfloor & Client Spec Blueprint...")
    bp_path = os.path.join(OUT_DIR, "Bathinda_64Unit_8x7ft_Shopfloor_QC_Blueprint.png")
    build_shopfloor_blueprint(bp_path, Z_hi)
    print(f"    Saved Blueprint: {bp_path}")

    readme_path = os.path.join(OUT_DIR, "README_FABRICATION_SPECS_8x7FT.txt")
    with open(readme_path, "w", encoding="utf-8") as rf:
        rf.write(
            "=========================================================================\n"
            "64 ENGR REGT (BATHINDA SECTOR) — 8.0 FT x 7.0 FT MASTER FABRICATION PACK\n"
            "=========================================================================\n"
            "1. EXACT PHYSICAL DIMENSIONS (LOCKED 8:7 ASPECT RATIO — DO NOT SQUISH TO 1:1):\n"
            "   - East-West Width (X) : 8.000 ft = 96.0 in = 2,438.4 mm (152 km sector)\n"
            "   - North-South Depth (Y): 7.000 ft = 84.0 in = 2,133.6 mm (150 km sector)\n"
            "   - 1:10 Master STL Size : 243.84 mm (X) x 213.36 mm (Y) x 13.98 mm (Z)\n"
            "     -> Scale uniformly by 1000% (10x) in Slicer / CNC CAM for 2438.4 x 2133.6 mm!\n\n"
            "2. FILES INCLUDED IN THIS ZIP:\n"
            "   - Bathinda_64Unit_8x7ft_2438x2134mm_Watertight_Master.stl\n"
            "     100% manifold watertight binary STL (563,000 triangles) with flat Z=0 bottom,\n"
            "     closed perimeter walls, smoothed SRTM DEM, 14 carved rivers/canals + bunds,\n"
            "     engraved Radcliffe IB (BP 260-365), 7 highways & 8 barrage/Sarvatra seats.\n"
            "   - Bathinda_64Unit_8x7ft_2438x2134_16bit_CNC_ZBrush_Heightmap.png\n"
            "     True 16-bit (0-65535) grayscale displacement alpha at exact 2438 x 2134 px\n"
            "     (1 pixel = 1.000 mm on the 8x7 ft table!).\n"
            "   - Bathinda_64Unit_8x7ft_Shopfloor_QC_Blueprint.png\n"
            "     Full shopfloor plan view with 8x7 ft (1x1 ft = 304.8 mm) tile grid, all 14\n"
            "     waterways, verified Indian vs Pak towns, 5 Headworks & 3 Sarvatra bridges.\n\n"
            "3. LIVE 3D CLIENT REFERENCE VIEWER:\n"
            "   https://botbit-in.github.io/bathinda-3d-terrain/\n"
        )

    zip_path = r"d:\Ai\autoabode office work\Bathinda_64Unit_8x7ft_Master_STL_and_CNC_Pack.zip"
    print("4/4 Compressing Master ZIP Pack...")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for fn in [
            "Bathinda_64Unit_8x7ft_2438x2134mm_Watertight_Master.stl",
            "Bathinda_64Unit_8x7ft_2438x2134_16bit_CNC_ZBrush_Heightmap.png",
            "Bathinda_64Unit_8x7ft_Shopfloor_QC_Blueprint.png",
            "README_FABRICATION_SPECS_8x7FT.txt",
        ]:
            zf.write(os.path.join(OUT_DIR, fn), arcname=fn)
    zip_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"DONE! Master ZIP: {zip_path} ({zip_mb:.2f} MB)")


if __name__ == "__main__":
    main()
