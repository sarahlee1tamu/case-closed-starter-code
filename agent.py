# agent.py
import os
import json
import math
import random
from typing import Dict, Any, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from flask import Flask, request, jsonify

_DEBUGGED_FIRST_STATE = False


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", "5008"))
MODEL_PATH = os.environ.get("MODEL_PATH", "training/models/dqn.pt")

print(f"[agent] Loading model from {MODEL_PATH}")

# Board defaults (used to size conv head if we load a conv checkpoint)
BOARD_W_DEFAULT = 20
BOARD_H_DEFAULT = 18

# Input feature size for the MLP encoder:
# [danger_up, danger_down, danger_left, danger_right]  (4)
# + onehot(current_dir)                                (4)
# + dx_to_opp_wrapped, dy_to_opp_wrapped               (2)
# + my_len_norm, opp_len_norm                          (2)
# + my_boosts_left_norm                                (1)
# + free_ratio                                         (1)
IN_DIM = 4 + 4 + 2 + 2 + 1 + 1  # = 14
NUM_ACTIONS = 4  # UP/DOWN/LEFT/RIGHT

app = Flask(__name__)

# --------------------------------------------------------------------------------------
# Model definitions
# --------------------------------------------------------------------------------------
class DQN(nn.Module):
    """MLP policy for 14-dim vector input."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, out_dim)
        )

    def forward(self, x):
        return self.net(x)

class ConvDQN(nn.Module):
    """Conv policy for (3, H, W) input; matches conv.* and head.* keys."""
    def __init__(self, H: int, W: int, out_dim: int):
        super().__init__()
        self.H, self.W = H, W
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
        )
        # NOTE: Include Flatten as the first module in 'head' so the Linear
        # layers are indexed at 1 and 3, matching your checkpoint keys.
        self.head = nn.Sequential(
            nn.Flatten(),                              # -> head.0 (no weights)
            nn.Linear(32 * H * W, 256),               # -> head.1.{weight,bias}
            nn.ReLU(),                                # -> head.2
            nn.Linear(256, out_dim),                  # -> head.3.{weight,bias}
        )

    def forward(self, x):                 # x: (B, 3, H, W)
        z = self.conv(x)                  # (B, 32, H, W)
        return self.head(z)               # head does Flatten internally


# --------------------------------------------------------------------------------------
# Utilities / helpers
# --------------------------------------------------------------------------------------
DIR2IDX = {"UP": 0, "DOWN": 1, "LEFT": 2, "RIGHT": 3}
IDX2DIR = ["UP", "DOWN", "LEFT", "RIGHT"]

ORDERED_DIRS = ["UP", "RIGHT", "DOWN", "LEFT"]
DIRS = {"UP": (0, -1), "RIGHT": (1, 0), "DOWN": (0, 1), "LEFT": (-1, 0)}
OPPOSITE = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}

def _safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        try:
            return int(float(x))
        except Exception:
            return default

def _xy(p) -> Tuple[int, int] | None:
    """Accept (x,y), [x,y], {'x':..,'y':..}, or 'x,y' -> (x,y) or None."""
    if p is None:
        return None
    # tuple/list
    if isinstance(p, (list, tuple)) and len(p) >= 2:
        return (_safe_int(p[0]), _safe_int(p[1]))
    # dict
    if isinstance(p, dict):
        if 'x' in p and 'y' in p:
            return (_safe_int(p['x']), _safe_int(p['y']))
        # Some judges use {'col':..,'row':..}
        if 'col' in p and 'row' in p:
            return (_safe_int(p['col']), _safe_int(p['row']))
    # string "x,y"
    if isinstance(p, str) and ',' in p:
        xs, ys = p.split(',', 1)
        return (_safe_int(xs), _safe_int(ys))
    return None

def _trail_points(tr) -> List[Tuple[int,int]]:
    """Normalize a trail list to a list of (x,y) tuples; skips bad items."""
    pts = []
    if isinstance(tr, list):
        for item in tr:
            xy = _xy(item)
            if xy is not None:
                pts.append(xy)
    return pts

def _infer_dir_from_pts(pts: List[Tuple[int,int]], W: int, H: int) -> str:
    if len(pts) >= 2:
        (x0, y0) = pts[-2]
        (x1, y1) = pts[-1]
        dx = _wrapped_step(_safe_int(x0), _safe_int(x1), W)
        dy = _wrapped_step(_safe_int(y0), _safe_int(y1), H)
        if abs(dx) >= abs(dy):
            return "RIGHT" if dx > 0 else ("LEFT" if dx < 0 else ("DOWN" if dy > 0 else "UP"))
        else:
            return "DOWN" if dy > 0 else "UP"
    return "RIGHT"


def _dir_from_steps(dx: int, dy: int) -> str:
    if abs(dx) > abs(dy):
        return "RIGHT" if dx > 0 else "LEFT"
    if abs(dy) > abs(dx):
        return "DOWN" if dy > 0 else "UP"
    # tie / zero → default
    return "RIGHT"

def _wrapped_step(a: int, b: int, size: int) -> int:
    """Smallest step from a->b on ring (returns -1,0,1 for 4-neighborhood)."""
    d = b - a
    # choose shorter way around torus
    if abs(d) > size // 2:
        d -= int(math.copysign(size, d))
    # clamp to {-1,0,1}
    return 0 if d == 0 else (1 if d > 0 else -1)


# === NEW: simple wrap helper (alias to your step_wrap) =======================
def wrap_xy(x, y, W, H):
    return (x % W, y % H)

# === NEW: occupied set from a 0/1 grid ======================================
def occ_from_grid(grid):
    H = len(grid)
    W = len(grid[0]) if H else 0
    occ = set()
    for y in range(H):
        for x in range(W):
            if grid[y][x]:  # or grid[y][x] != '.'
                occ.add((x, y))
    return occ, W, H

# === NEW: occupied set from explicit trails (permanent walls) ================
# Expects me["trail"] and opp["trail"] to be lists of (x,y) INCLUDING heads.
def occ_from_trails(me, opp, W, H):
    occ = set()
    for (x, y) in (me.get("trail") or []):
        occ.add(wrap_xy(x, y, W, H))
    for (x, y) in (opp.get("trail") or []):
        occ.add(wrap_xy(x, y, W, H))
    return occ

# === NEW: membership check with wrap =========================================
def occupied_wrap(occ_set, x, y, W, H):
    return wrap_xy(x, y, W, H) in occ_set


def step_wrap(x, y, dx, dy, W, H):
    return (x + dx) % W, (y + dy) % H

def run_length(pos, dir_str, occ_set, W, H, max_steps=512):
    x, y = pos
    dx, dy = DIRS[dir_str]
    cnt = 0
    for _ in range(max_steps):
        x, y = step_wrap(x, y, dx, dy, W, H)
        if (x, y) in occ_set:
            break
        cnt += 1
    return cnt

def best_safe_dir(head, cur_dir, occ_set, W, H):
    # Score: (run_len, tie-break by fixed order)
    candidates = [d for d in ORDERED_DIRS if d != OPPOSITE.get(cur_dir)]
    scored = []
    for d in candidates:
        nx, ny = step_wrap(head[0], head[1], *DIRS[d], W, H)
        if (nx, ny) in occ_set:
            continue
        rl = run_length(head, d, occ_set, W, H)
        nx2, ny2 = step_wrap(nx, ny, *DIRS[d], W, H)
        trap = 1 if (nx2, ny2) in occ_set else 0
        scored.append((rl - 5 * trap, -ORDERED_DIRS.index(d), d))
    if not scored:
        return cur_dir  # nothing safe; send something non-reverse
    scored.sort(reverse=True)
    return scored[0][2]

def _get(obj: Dict, *keys, default=None):
    """Safely read nested keys."""
    cur = obj
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def torus_delta(a: int, b: int, size: int) -> int:
    """Smallest signed distance from a to b on a ring of given size."""
    d = b - a
    if abs(d) > size // 2:
        d = d - int(math.copysign(size, d))
    return d

def coerce_bool_grid(occupied) -> List[List[int]]:
    """Coerce any reasonable incoming occupancy representation into 2D 0/1 list."""
    if isinstance(occupied, list) and occupied and isinstance(occupied[0], list):
        H = len(occupied)
        W = len(occupied[0])
        grid = [[1 if occupied[y][x] else 0 for x in range(W)] for y in range(H)]
        return grid
    return occupied  # will be handled later when width/height are known

def build_occupied_grid(state: Dict[str, Any], W: int, H: int) -> List[List[int]]:
    """
    Accepts any of:
      - state["occupied"] as HxW 0/1 or list of {x,y}
      - state["grid"]            (HxW, 0/1/char) OR state["board"]["grid"]
      - infer from 'agents'/'players'/('p1','p2','agent1','agent2') trails
    """
    # 1) Direct occupied boolean grid
    occ = state.get("occupied")
    if isinstance(occ, list) and occ and isinstance(occ[0], list):
        return coerce_bool_grid(occ)

    # 2) Pair-list occupied
    if isinstance(occ, list) and (len(occ) == 0 or isinstance(occ[0], dict)):
        grid = [[0 for _ in range(W)] for _ in range(H)]
        for c in occ:
            x = _get(c, "x", default=None)
            y = _get(c, "y", default=None)
            if x is not None and y is not None:
                grid[y % H][x % W] = 1
        return grid

    # 3) Look for a grid (top-level or under board)
    grid_src = state.get("grid")
    if grid_src is None:
        board = state.get("board") or {}
        grid_src = board.get("grid")
    if isinstance(grid_src, list) and grid_src:
        # Accept 0/1 ints OR chars like '.'/'A' etc.
        H2 = len(grid_src); W2 = len(grid_src[0])
        out = [[0 for _ in range(W)] for _ in range(H)]
        for y in range(min(H, H2)):
            for x in range(min(W, W2)):
                cell = grid_src[y][x]
                out[y][x] = 1 if (cell and cell != 0 and cell != "." and cell != " ") else 0
        return out

    # 4) Infer from trails/agents
    grid = [[0 for _ in range(W)] for _ in range(H)]

    def mark_xy(x, y):
        grid[y % H][x % W] = 1

    # Common buckets where the judge might put actors
    buckets = []
    for k in ["agents", "players", "snakes", "bikes", "units"]:
        arr = state.get(k)
        if isinstance(arr, list):
            buckets.append(arr)

    # Named players, e.g., p1/p2, agent1/agent2
    for k in ["p1", "p2", "agent1", "agent2", "player1", "player2"]:
        a = state.get(k)
        if isinstance(a, dict):
            buckets.append([a])

    # Explicit singletons possibly present:
    for k in ["me", "self", "agent", "player", "opponent", "enemy"]:
        a = state.get(k)
        if isinstance(a, dict):
            buckets.append([a])

    for arr in buckets:
        for ent in arr:
            hx = _get(ent, "x", default=_get(ent, "pos", "x", default=None))
            hy = _get(ent, "y", default=_get(ent, "pos", "y", default=None))
            if hx is not None and hy is not None:
                mark_xy(int(hx), int(hy))
            trail = ent.get("trail") or ent.get("body") or ent.get("segments")
            if isinstance(trail, list):
                for seg in trail:
                    # support (x,y), {"x":..,"y":..}, or [x,y]
                    if isinstance(seg, (tuple, list)) and len(seg) >= 2:
                        tx, ty = int(seg[0]), int(seg[1])
                    else:
                        tx = _get(seg, "x", default=None)
                        ty = _get(seg, "y", default=None)
                        if tx is None or ty is None:
                            continue
                        tx, ty = int(tx), int(ty)
                    mark_xy(tx, ty)

    return grid


def extract_agents(state: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Return (me, opp) unified dicts with keys: x,y,dir,length,boosts_left.
    Uses state['player_number'] if present to decide who is 'me' when multiple players exist.
    Handles diverse shapes (agents/players arrays, p1/p2, agent1/agent2, etc.)
    """
    # If the judge passes which player we are (1 or 2)
    me_id = state.get("player_number") or state.get("you") or state.get("player_id")

    # Direct me/opp
    me = state.get("me") or state.get("self") or state.get("agent") or state.get("player") or {}
    opp = state.get("opponent") or state.get("enemy") or {}

    # Arrays
    arrays = []
    for k in ["agents", "players", "snakes", "bikes", "units"]:
        arr = state.get(k)
        if isinstance(arr, list):
            arrays.append(arr)

    # Named singletons
    named = []
    for k in ["p1", "p2", "agent1", "agent2", "player1", "player2"]:
        a = state.get(k)
        if isinstance(a, dict):
            named.append((k, a))

    # If me/opp missing, try infer from arrays/named + me_id
    if (not me) or (not opp):
        candidates: List[Dict[str, Any]] = []
        for arr in arrays:
            candidates.extend(arr)
        for k, a in named:
            # attach a simple id if not present
            if "id" not in a:
                a = dict(a)
                a["id"] = 1 if ("1" in k or "p1" in k) else (2 if ("2" in k or "p2" in k) else k)
            candidates.append(a)

        # select 'me' by id if possible
        if candidates:
            m = None
            if me_id is not None:
                for a in candidates:
                    aid = a.get("id") or a.get("player_number") or a.get("idx")
                    if str(aid) == str(me_id):
                        m = a
                        break
            if m is None:
                # fallback: first is me, second is opp
                m = candidates[0]
            # choose an opponent distinct from me
            o = None
            for a in candidates:
                if a is not m:
                    o = a
                    break
            me = me or m or {}
            opp = opp or o or {}

    def norm_one(a: Dict[str, Any]) -> Dict[str, Any]:
        x = _get(a, "x", default=_get(a, "pos", "x", default=_get(a, "head", "x", default=0)))
        y = _get(a, "y", default=_get(a, "pos", "y", default=_get(a, "head", "y", default=0)))
        direction = a.get("dir") or a.get("direction") or a.get("facing") or "RIGHT"
        trail = a.get("trail") or a.get("body") or a.get("segments") or []
        length = a.get("length") or (len(trail) if isinstance(trail, list) else 1) or 1
        boosts = a.get("boosts_left") or a.get("boosts") or 0
        return {"x": int(x), "y": int(y), "dir": str(direction), "length": int(length), "boosts_left": int(boosts)}

    return norm_one(me), norm_one(opp)


def parse_state(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalizes different judge formats into our canonical observation.
    Special-case: flat schema with agent1_/agent2_ fields and player_number.
    """
       # ----------- NEW: flat judge schema (your case) -----------
    if "agent1_trail" in payload and "agent2_trail" in payload:
        board = payload.get("board", {}) or {}

        # Determine W,H from various board shapes
        if isinstance(board, dict):
            grid_src = board.get("grid")
            if isinstance(grid_src, list) and grid_src:
                H = _safe_int(len(grid_src), BOARD_H_DEFAULT)
                W = _safe_int(len(grid_src[0]) if H else BOARD_W_DEFAULT, BOARD_W_DEFAULT)
            else:
                W = _safe_int(board.get("width", board.get("W", BOARD_W_DEFAULT)), BOARD_W_DEFAULT)
                H = _safe_int(board.get("height", board.get("H", BOARD_H_DEFAULT)), BOARD_H_DEFAULT)
        elif isinstance(board, list) and board:  # board is actually a grid
            H = _safe_int(len(board), BOARD_H_DEFAULT)
            W = _safe_int(len(board[0]) if H else BOARD_W_DEFAULT, BOARD_W_DEFAULT)
        else:
            W, H = BOARD_W_DEFAULT, BOARD_H_DEFAULT

        # Normalize trails
        t1_pts = _trail_points(payload.get("agent1_trail", []) or [])
        t2_pts = _trail_points(payload.get("agent2_trail", []) or [])

        # Build occupancy from permanent trails
        occ = [[0 for _ in range(W)] for _ in range(H)]
        for (x, y) in t1_pts:
            occ[y % H][x % W] = 1
        for (x, y) in t2_pts:
            occ[y % H][x % W] = 1

        # Heads (last points if any, else 0,0)
        h1x, h1y = (_safe_int(t1_pts[-1][0]) % W, _safe_int(t1_pts[-1][1]) % H) if t1_pts else (0, 0)
        h2x, h2y = (_safe_int(t2_pts[-1][0]) % W, _safe_int(t2_pts[-1][1]) % H) if t2_pts else (0, 0)

        # Directions
        d1 = _infer_dir_from_pts(t1_pts, W, H)
        d2 = _infer_dir_from_pts(t2_pts, W, H)

        a1 = {
            "x": h1x, "y": h1y, "dir": d1,
            "length": _safe_int(payload.get("agent1_length", max(1, len(t1_pts))), max(1, len(t1_pts))),
            "boosts_left": _safe_int(payload.get("agent1_boosts", 0), 0),
            "alive": bool(payload.get("agent1_alive", True)),
        }
        a2 = {
            "x": h2x, "y": h2y, "dir": d2,
            "length": _safe_int(payload.get("agent2_length", max(1, len(t2_pts))), max(1, len(t2_pts))),
            "boosts_left": _safe_int(payload.get("agent2_boosts", 0), 0),
            "alive": bool(payload.get("agent2_alive", True)),
        }

        me_num = str(payload.get("player_number", "1"))
        me, opp = (a1, a2) if me_num == "1" else (a2, a1)

        occ_count = sum(sum(row) for row in occ)
        total = W * H
        free_ratio = max(0.0, min(1.0, 1.0 - occ_count / float(total))) if total > 0 else 0.5

        return {
            "board_width": W,
            "board_height": H,
            "occupied": occ,
            "me": me,
            "opponent": opp,
            "free_ratio": free_ratio
        }


    # ----------- FALLBACK: your previous generic parser -----------
    state = payload.get("state", payload)
    W = state.get("board_width") or state.get("width") or _get(state, "board", "width")
    H = state.get("board_height") or state.get("height") or _get(state, "board", "height")

    grid_src = state.get("grid")
    if grid_src is None:
        grid_src = _get(state, "board", "grid")
    if (W is None or H is None) and isinstance(grid_src, list) and grid_src:
        H = H or len(grid_src)
        W = W or (len(grid_src[0]) if H else None)

    W = int(W or BOARD_W_DEFAULT)
    H = int(H or BOARD_H_DEFAULT)

    me, opp = extract_agents(state)
    occupied = build_occupied_grid(state, W, H)

    occ_count = sum(sum(row) for row in occupied)
    total = W * H
    free_ratio = max(0.0, min(1.0, 1.0 - occ_count / float(total))) if total > 0 else 0.5

    return {
        "board_width": W,
        "board_height": H,
        "occupied": occupied,
        "me": me,
        "opponent": opp,
        "free_ratio": free_ratio
    }



# --------------------------------------------------------------------------------------
# Feature encoders + safety checks
# --------------------------------------------------------------------------------------
def cell_occupied(occ_grid: List[List[int]], W: int, H: int, x: int, y: int) -> bool:
    return occ_grid[y % H][x % W] == 1

def next_xy(x: int, y: int, dir_str: str) -> Tuple[int, int]:
    if dir_str == "UP":    return x, y - 1
    if dir_str == "DOWN":  return x, y + 1
    if dir_str == "LEFT":  return x - 1, y
    return x + 1, y  # RIGHT

def encode_state(obs: Dict[str, Any]) -> np.ndarray:
    """14-dim vector encoder for MLP."""
    W, H = obs["board_width"], obs["board_height"]
    occ = obs["occupied"]
    me, opp = obs["me"], obs["opponent"]

    myx, myy = me["x"], me["y"]
    # Dangers
    cand_dirs = ["UP", "DOWN", "LEFT", "RIGHT"]
    danger = []
    for d in cand_dirs:
        nx, ny = next_xy(myx, myy, d)
        danger.append(1.0 if cell_occupied(occ, W, H, nx, ny) else 0.0)

    # One-hot current dir
    dir_idx = DIR2IDX.get(me["dir"], 3)
    onehot_dir = [1.0 if i == dir_idx else 0.0 for i in range(4)]

    # Wrapped deltas
    dx = torus_delta(myx, opp["x"], W) / (W / 2.0 if W > 1 else 1.0)
    dy = torus_delta(myy, opp["y"], H) / (H / 2.0 if H > 1 else 1.0)

    my_len = float(me["length"]) / 200.0
    opp_len = float(opp["length"]) / 200.0
    boosts = float(me["boosts_left"]) / 3.0
    free_ratio = float(obs.get("free_ratio", 0.5))

    vec = np.array(danger + onehot_dir + [dx, dy, my_len, opp_len, boosts, free_ratio], dtype=np.float32)
    return vec

def encode_state_conv(obs: Dict[str, Any]) -> np.ndarray:
    """(3, H, W) encoder for Conv policy: [occupied, me_head, opp_head]."""
    W, H = obs["board_width"], obs["board_height"]
    occ = np.array(obs["occupied"], dtype=np.float32)  # (H, W)

    me = obs["me"];  opp = obs["opponent"]
    me_mask  = np.zeros((H, W), dtype=np.float32)
    opp_mask = np.zeros((H, W), dtype=np.float32)
    me_mask[me["y"] % H, me["x"] % W]   = 1.0
    opp_mask[opp["y"] % H, opp["x"] % W] = 1.0

    x = np.stack([occ, me_mask, opp_mask], axis=0)     # (3, H, W)
    return x

def valid_action_mask(obs: Dict[str, Any]) -> np.ndarray:
    W, H = obs["board_width"], obs["board_height"]
    occ = obs["occupied"]
    me = obs["me"]
    myx, myy = me["x"], me["y"]

    mask = []
    for d in ["UP", "DOWN", "LEFT", "RIGHT"]:
        nx, ny = next_xy(myx, myy, d)
        mask.append(0.0 if cell_occupied(occ, W, H, nx, ny) else 1.0)
    return np.array(mask, dtype=np.float32)

def two_step_safe(obs: Dict[str, Any], dir_str: str) -> bool:
    """Quick 2-step safety check for BOOST."""
    W, H = obs["board_width"], obs["board_height"]
    occ = obs["occupied"]
    me = obs["me"]
    x1, y1 = next_xy(me["x"], me["y"], dir_str)
    if cell_occupied(occ, W, H, x1, y1):
        return False
    x2, y2 = next_xy(x1, y1, dir_str)
    if cell_occupied(occ, W, H, x2, y2):
        return False
    return True

def occ_grid_to_set(occ_grid: List[List[int]]) -> set[tuple[int, int]]:
    H = len(occ_grid)
    W = len(occ_grid[0]) if H else 0
    return {(x, y) for y in range(H) for x in range(W) if occ_grid[y][x] == 1}

def ensure_sane_dir(base_dir: str, obs: Dict[str, Any]) -> str:
    """
    If the proposed direction would immediately collide (including wrap),
    or creates an obvious 2-step trap, pivot to a safer direction.
    """
    W, H = obs["board_width"], obs["board_height"]
    occ = obs["occupied"]
    me = obs["me"]
    head = (me["x"], me["y"])
    cur_dir = me.get("dir", "RIGHT")

    occ_set = occ_grid_to_set(occ)

    # 1) Immediate collision check (wrap)
    nx, ny = next_xy(head[0], head[1], base_dir)
    nx, ny = wrap_xy(nx, ny, W, H)
    if (nx, ny) in occ_set:
        return best_safe_dir(head, cur_dir, occ_set, W, H)

    # 2) Two-step trap check
    nnx, nny = next_xy(nx, ny, base_dir)
    nnx, nny = wrap_xy(nnx, nny, W, H)
    if (nnx, nny) in occ_set:
        alt = best_safe_dir(head, cur_dir, occ_set, W, H)
        ax, ay = next_xy(head[0], head[1], alt)
        ax, ay = wrap_xy(ax, ay, W, H)
        if (ax, ay) not in occ_set:
            return alt

    return base_dir


# --------------------------------------------------------------------------------------
# Model loading (robust to conv vs mlp)
# --------------------------------------------------------------------------------------
IS_CONV = False
policy: nn.Module = DQN(IN_DIM, NUM_ACTIONS)  # placeholder; will be replaced on load

def try_load_policy(path: str) -> bool:
    global IS_CONV, policy
    if not os.path.exists(path):
        print(f"[agent] Model not found at {path}; running heuristic only.")
        policy = DQN(IN_DIM, NUM_ACTIONS)
        return False
    try:
        sd = torch.load(path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        keys = list(sd.keys())
        looks_conv = any(k.startswith("conv.") for k in keys) and any(k.startswith("head.") for k in keys)
        if looks_conv:
            IS_CONV = True
            print("[agent] Detected CONV checkpoint; instantiating ConvDQN(3xHxW).")
            policy = ConvDQN(BOARD_H_DEFAULT, BOARD_W_DEFAULT, NUM_ACTIONS)
        else:
            IS_CONV = False
            print("[agent] Detected MLP checkpoint; instantiating DQN(14->4).")
            policy = DQN(IN_DIM, NUM_ACTIONS)

        missing, unexpected = policy.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"[agent] Loaded with strict=False (missing={missing}, unexpected={unexpected}).")
        else:
            print("[agent] Model loaded cleanly.")
        return True
    except Exception as e:
        print(f"[agent] Could not load model ({e}); running heuristic only.")
        policy = DQN(IN_DIM, NUM_ACTIONS)
        return False

MODEL_LOADED = try_load_policy(MODEL_PATH)
policy.eval()

# --------------------------------------------------------------------------------------
# Action selection
# --------------------------------------------------------------------------------------
def pick_action(obs: Dict[str, Any]) -> str:
    """
    Choose UP/DOWN/LEFT/RIGHT (optionally with :BOOST) with a hard safety veto
    that avoids immediate collisions (including wrap) and simple 2-step traps.
    """
    W, H = obs["board_width"], obs["board_height"]
    occ_grid = obs["occupied"]
    me = obs["me"]; opp = obs["opponent"]

    mask = valid_action_mask(obs)  # 1=safe, 0=unsafe

    # --- Heuristic base (fallback if no model / model blocked) ---
    if np.sum(mask) == 0:
        base_dir = random.choice(IDX2DIR)
    else:
        cur_idx = DIR2IDX.get(me["dir"], 3)
        if mask[cur_idx] > 0.5:
            base_dir = IDX2DIR[cur_idx]       # keep going if safe
        else:
            safe_idxs = np.where(mask > 0.5)[0]
            base_dir = IDX2DIR[int(np.random.choice(safe_idxs))]

    # --- Model ranking (if available), masked by immediate safety ---
    try:
        with torch.no_grad():
            if IS_CONV:
                x = encode_state_conv(obs)                     # (3, H, W)
                s = torch.from_numpy(x).unsqueeze(0).float()   # (1, 3, H, W)
            else:
                x = encode_state(obs)                          # (14,)
                s = torch.from_numpy(x).unsqueeze(0).float()   # (1, 14)
            q = policy(s).squeeze(0).numpy()
        q[mask < 0.5] = -1e9
        if np.any(mask > 0.5):
            base_dir = IDX2DIR[int(np.argmax(q))]
    except Exception:
        pass  # ignore model issues; stick with heuristic base_dir

    # --- FINAL SAFETY VETO (wrap + trails as permanent walls) ---
    base_dir = ensure_sane_dir(base_dir, obs)

    # --- BOOST only if two steps ahead are clear in the *final* direction ---
    do_boost = False
    if me.get("boosts_left", 0) > 0:
        longer = me.get("length", 1) >= opp.get("length", 1)
        want = longer or (random.random() < 0.10)
        if want and two_step_safe(obs, base_dir):
            do_boost = True

    move = base_dir + (":BOOST" if do_boost else "")

    # --- NEW: tiny debug print (counts occupied cells; great sanity check) ---
    if __debug__:
        occ_count = sum(sum(row) for row in occ_grid)
        head = (me["x"], me["y"])
        print(f"[dbg] head={head} dir={me.get('dir','RIGHT')} choice={base_dir} occ_count={occ_count}")

    return move





# --------------------------------------------------------------------------------------
# Flask endpoints
# --------------------------------------------------------------------------------------
_last_state = None

@app.route("/", methods=["GET"])
def home():
    # Required keys for local-tester
    return jsonify({
        "ok": True,
        "participant": "Participant1",
        "agent_name": "Agent1",
        "model_loaded": MODEL_LOADED,
        "model_type": "conv" if IS_CONV else "mlp"
    })

@app.route("/send-state", methods=["POST"])
def send_state():
    global _last_state, _DEBUGGED_FIRST_STATE
    data = request.get_json(force=True) or {}
    try:
        if not _DEBUGGED_FIRST_STATE:
            try:
                print("[state] top-level keys:", list(data.keys()))
                st = data.get("state", {})
                if st:
                    print("[state.state] keys:", list(st.keys()))
                bd = (st.get("board") if isinstance(st, dict) else None) or data.get("board")
                if bd:
                    if isinstance(bd, dict):
                        print("[state.board] keys:", list(bd.keys()))
                    elif isinstance(bd, list):
                        print("[state.board] is a grid: H=", len(bd), " W=", (len(bd[0]) if bd else 0))
                preview = json.dumps(data)
                print("[state] preview:", (preview[:1500] + ("..." if len(preview) > 1500 else "")))
            except Exception:
                pass
            _DEBUGGED_FIRST_STATE = True

        _last_state = parse_state(data)
        return jsonify({"ok": True})
    except Exception as e:
        # NEW: print the error so we see why the 400 happened
        print("[send-state][error]", repr(e))
        return jsonify({"ok": False, "error": str(e)}), 400



def is_opposite(cur_dir: str, next_dir: str) -> bool:
    return OPPOSITE.get(cur_dir) == next_dir

@app.route("/send-move", methods=["GET", "POST"])
def send_move():
    """
    Tester may call GET /send-move with no body.
    Judge may call POST /send-move with a body.
    We handle both. If no state in the request, fall back to _last_state.
    """
    global _last_state

    data = {}
    if request.method == "POST":
        try:
            data = request.get_json(force=True) or {}
        except Exception:
            data = {}

    try:
        if "state" in data or "board_width" in data or "width" in data:
            obs = parse_state(data)
        else:
            obs = _last_state
    except Exception:
        obs = _last_state

    if obs is None:
        return jsonify({"move": "RIGHT"})  # default; judge will accept

    move = pick_action(obs)

    # Extract base direction / boost intent
    cur_dir = obs["me"].get("dir", "RIGHT")
    base_dir = move.split(":")[0]
    want_boost = move.endswith(":BOOST")

    # 1) Never allow exact reverse (engine may mark invalid)
    if is_opposite(cur_dir, base_dir):
        mask = valid_action_mask(obs)
        cur_idx = DIR2IDX.get(cur_dir, 3)
        if mask[cur_idx] > 0.5:
            base_dir = IDX2DIR[cur_idx]
        else:
            safe_idxs = np.where(mask > 0.5)[0]
            if len(safe_idxs) > 0:
                base_dir = IDX2DIR[int(np.random.choice(safe_idxs))]
            else:
                base_dir = cur_dir  # nothing safe; let engine handle
        want_boost = want_boost and two_step_safe(obs, base_dir)

    # 2) Final safety veto (wrap/self/any collision + simple 2-step trap check)
    base_dir = ensure_sane_dir(base_dir, obs)

    # 3) Keep BOOST only if still safe for two steps
    if want_boost and not two_step_safe(obs, base_dir):
        want_boost = False

    move = base_dir + (":BOOST" if want_boost else "")
    return jsonify({"move": move})

@app.route("/end", methods=["POST"])
def end():
    return jsonify({"ok": True})

if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=False)
