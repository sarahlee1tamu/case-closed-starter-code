"""
Smarter sample_agent for Case Closed – Flask server that follows the Judge Protocol.
- Avoids immediate crashes (your trail, opponent trail, board walls via torus wrap handled).
- Chooses the move that maximizes reachable open space (1-step lookahead flood-fill).
- Dodges head-on collisions when disadvantaged by length.
- Uses BOOST sparingly: when we're funneled (only one safe move) or when boost increases safe space.
"""
import os
from typing import List, Tuple, Dict, Any, Set
from collections import deque

from flask import Flask, request, jsonify

app = Flask(__name__)

# Identity (used by local tools/judge UI)
PARTICIPANT = os.getenv("PARTICIPANT", "SampleParticipant")
AGENT_NAME  = os.getenv("AGENT_NAME",  "SampleAgentPlus")

# -----------------------------------------------------------------------------
#State cache (the judge POSTs /send-state each turn before /send-move)
# -----------------------------------------------------------------------------
game_state: Dict[str, Any] = {}

# Grid helpers
DIRS = {
    "UP":    (0, -1),
    "DOWN":  (0,  1),
    "LEFT":  (-1, 0),
    "RIGHT": (1,  0),
}
ORDERED_DIRS = ["UP", "RIGHT", "DOWN", "LEFT"]  # tie-break order (clockwise)
OPPOSITE = {"UP":"DOWN","DOWN":"UP","LEFT":"RIGHT","RIGHT":"LEFT"}


def dims(board: List[List[int]]) -> Tuple[int,int]:
    H = len(board)
    W = len(board[0]) if H else 0
    return W, H


def wrap(x: int, y: int, W: int, H: int) -> Tuple[int,int]:
    return x % W, y % H


def step(pos: Tuple[int,int], dir_str: str, W: int, H: int) -> Tuple[int,int]:
    dx, dy = DIRS[dir_str]
    return wrap(pos[0] + dx, pos[1] + dy, W, H)


def occupied_from_board(board: List[List[int]]) -> Set[Tuple[int,int]]:
    occ = set()
    H = len(board)
    for y in range(H):
        row = board[y]
        for x, v in enumerate(row):
            if v == 1:  # AGENT cells
                occ.add((x,y))
    return occ


def reachable_size(start: Tuple[int,int], occ: Set[Tuple[int,int]], W:int, H:int, limit:int=400) -> int:
    """Flood fill counting free cells reachable from start (up to an upper limit to keep it fast)."""
    if start in occ:
        return 0
    q = deque([start])
    seen = {start}
    count = 0
    while q and count < limit:
        x,y = q.popleft()
        count += 1
        for dir_str in ORDERED_DIRS:
            nx, ny = step((x,y), dir_str, W, H)
            if (nx,ny) not in occ and (nx,ny) not in seen:
                seen.add((nx,ny))
                q.append((nx,ny))
    return count


def head_on_is_bad(my_len:int, opp_len:int) -> bool:
    """Return True if a head-on tie would be bad for us (we lose ties or when shorter)."""
    # In this competition build: longer trail survives head-on; equal length often kills both.
    # Treat tie as bad/risky for the sample bot.
    return opp_len >= my_len


def choose_dir_safe(board, my_trail, opp_trail, my_len, opp_len, cur_dir, boosts_left) -> Tuple[str,bool]:
    """Pick a direction and whether to use BOOST."""
    if not my_trail:
        return "RIGHT", False

    W,H = dims(board)
    occ = occupied_from_board(board)

    my_head = my_trail[-1]
    opp_head = opp_trail[-1] if opp_trail else None

    # Build candidate directions (avoid instant self-reverse; judge would correct it, but we filter anyway)
    candidates = [d for d in ORDERED_DIRS if d != OPPOSITE.get(cur_dir)]
    # Filter: cannot move into occupied cell on the first step
    first_step_ok = []
    for d in candidates:
        n1 = step(my_head, d, W, H)
        if n1 in occ:
            continue
        # Head-on avoidance: if opponent is adjacent and moving into their head cell would be a tie/bad, avoid
        if opp_head and n1 == opp_head and head_on_is_bad(my_len, opp_len):
            continue
        first_step_ok.append(d)

    if not first_step_ok:
        # If everything is blocked, fall back to any candidate (judge will handle invalid opposite)
        return candidates[0], False

    # Score each by reachable space after moving there.
    scored = []
    for d in first_step_ok:
        n1 = step(my_head, d, W, H)
        occ1 = set(occ)
        occ1.add(n1)  # our trail occupies after we move
        space1 = reachable_size(n1, occ1, W, H)

        # Optional: estimate benefit of BOOST (two steps) only if we have boosts
        boost_gain = 0
        if boosts_left > 0:
            n2 = step(n1, d, W, H)
            if n2 not in occ1:
                occ2 = set(occ1)
                occ2.add(n2)
                space2 = reachable_size(n2, occ2, W, H)
                boost_gain = max(0, space2 - space1)  # added safe space by stepping twice

        # Distance from opponent head (prefer farther when close)
        dist_opp = 0
        if opp_head:
            ox, oy = opp_head
            dx = min((n1[0]-ox) % W, (ox-n1[0]) % W)
            dy = min((n1[1]-oy) % H, (oy-n1[1]) % H)
            dist_opp = dx + dy

        scored.append((space1, boost_gain, dist_opp, -ORDERED_DIRS.index(d), d))

    # Choose by max tuple (space, boost_gain, dist_opp, tiebreak by fixed order)
    scored.sort(reverse=True)
    best_space, best_boost_gain, _, _, best_dir = scored[0]

    # Use BOOST if:
    # - we only have one safe move (escaping a corridor), OR
    # - boost increases reachable space noticeably
    use_boost = False
    if boosts_left > 0:
        if len(first_step_ok) == 1:
            use_boost = True
        elif best_boost_gain >= 6:  # heuristic threshold
            use_boost = True

    return best_dir, use_boost


@app.route("/", methods=["GET"])
def index():
    return jsonify({"participant": PARTICIPANT, "agent_name": AGENT_NAME}), 200


@app.route("/send-state", methods=["POST"])
def send_state():
    data = request.get_json(silent=True) or {}
    # Keep only expected keys; tolerate extras
    game_state.clear()
    game_state.update(data)
    return jsonify({"status": "ok"}), 200


@app.route("/send-move", methods=["GET"])
def send_move():
    # Pull state pushed on previous /send-state
    board = game_state.get("board", [])
    pnum  = request.args.get("player_number", type=int) or game_state.get("player_number", 1)
    turn  = request.args.get("turn_count", type=int) or game_state.get("turn_count", 0)

    if pnum == 1:
        my_trail   = game_state.get("agent1_trail", [])
        opp_trail  = game_state.get("agent2_trail", [])
        my_len     = game_state.get("agent1_length", 1)
        opp_len    = game_state.get("agent2_length", 1)
        my_boosts  = game_state.get("agent1_boosts", 3)
    else:
        my_trail   = game_state.get("agent2_trail", [])
        opp_trail  = game_state.get("agent1_trail", [])
        my_len     = game_state.get("agent2_length", 1)
        opp_len    = game_state.get("agent1_length", 1)
        my_boosts  = game_state.get("agent2_boosts", 3)

    # Determine current direction from last two trail points if available; default RIGHT
    cur_dir = "RIGHT"
    if len(my_trail) >= 2:
        (x1,y1),(x2,y2) = my_trail[-2], my_trail[-1]
        W,H = dims(board) if board else (20,18)
        # Choose the non-wrapped delta (torus shortest move)
        dx = (x2 - x1 + W) % W
        dy = (y2 - y1 + H) % H
        if dx == 0 and dy == 0:
            pass
        else:
            if dx == 1 or dx == (W-1):
                cur_dir = "RIGHT" if dx == 1 else "LEFT"
            elif dy == 1 or dy == (H-1):
                cur_dir = "DOWN" if dy == 1 else "UP"

    move_dir, want_boost = choose_dir_safe(board, my_trail, opp_trail, my_len, opp_len, cur_dir, my_boosts)

    return jsonify({"move": f"{move_dir}:BOOST" if want_boost else move_dir}), 200


@app.route("/end", methods=["POST"])
def end():
    # Optional place to log results/reset
    return jsonify({"ok": True}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5009"))
    print(f"[{AGENT_NAME}] starting on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
