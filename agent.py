import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, jsonify
from threading import Lock
from collections import deque

from case_closed_game import Game, Direction, GameResult

# ----- Flask app FIRST -----
app = Flask(__name__)

GLOBAL_GAME = Game()
LAST_POSTED_STATE = {}
game_lock = Lock()

PARTICIPANT = "RLTeam"
AGENT_NAME = "TronRL"

device = torch.device("cpu")

GRID_WIDTH = 20
GRID_HEIGHT = 18

class DQN(nn.Module):
    def __init__(self, state_size=128, action_size=4):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, action_size)
    def forward(self, x):
        x = F.relu(self.fc1(x)); x = F.relu(self.fc2(x)); x = F.relu(self.fc3(x))
        return self.fc4(x)

def in_bounds(r, c, rows=GRID_HEIGHT, cols=GRID_WIDTH): return 0 <= r < rows and 0 <= c < cols
def is_safe(r, c, board, rows=GRID_HEIGHT, cols=GRID_WIDTH): return in_bounds(r, c, rows, cols) and board[r][c] == 0

def flood_fill_size(sr, sc, board, rows=GRID_HEIGHT, cols=GRID_WIDTH, limit=None):
    if not in_bounds(sr, sc, rows, cols) or board[sr][sc] != 0: return 0
    seen = [[False]*cols for _ in range(rows)]
    q = deque([(sr, sc)]); seen[sr][sc] = True
    size = 0; lim = limit or (rows * cols)
    for_d = [(0,-1),(0,1),(-1,0),(1,0)]
    while q and size < lim:
        r, c = q.popleft(); size += 1
        for dr, dc in for_d:
            nr, nc = r+dr, c+dc
            if in_bounds(nr, nc, rows, cols) and not seen[nr][nc] and board[nr][nc] == 0:
                seen[nr][nc] = True; q.append((nr, nc))
    return size

def count_open_neighbors(r, c, board, rows=GRID_HEIGHT, cols=GRID_WIDTH):
    cnt = 0
    for dr, dc in [(0,-1),(0,1),(-1,0),(1,0)]:
        nr, nc = r+dr, c+dc
        if in_bounds(nr, nc, rows, cols) and board[nr][nc] == 0: cnt += 1
    return cnt

def manhattan_distance(r1, c1, r2, c2): return abs(r1-r2)+abs(c1-c2)

class TronAgent:
    def __init__(self, model_path='model.pth'):
        self.model = DQN().to(device)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.model.eval()
            print(f"Loaded model from {model_path}")
        self.action_str_map = {0:"UP",1:"DOWN",2:"LEFT",3:"RIGHT"}

    def extract_features(self, state, player_num):
        board = state.get("board", [[0]*GRID_WIDTH for _ in range(GRID_HEIGHT)])
        if player_num == 1:
            my_trail = state.get("agent1_trail", [])
            my_boosts = state.get("agent1_boosts", 0)
            other_trail = state.get("agent2_trail", [])
            other_boosts = state.get("agent2_boosts", 0)
        else:
            my_trail = state.get("agent2_trail", [])
            my_boosts = state.get("agent2_boosts", 0)
            other_trail = state.get("agent1_trail", [])
            other_boosts = state.get("agent1_boosts", 0)

        if not my_trail:
            return np.zeros(128, dtype=np.float32)

        # heads: (x=col, y=row)
        head_x, head_y = my_trail[-1]
        head_c, head_r = head_x, head_y  # column, row

        # normalized position
        norm_r = head_r / GRID_HEIGHT
        norm_c = head_c / GRID_WIDTH

        # direction vector (rows/cols deltas)
        if len(my_trail) >= 2:
            prev_x, prev_y = my_trail[-2]
            dr = (head_y - prev_y) / GRID_HEIGHT
            dc = (head_x - prev_x) / GRID_WIDTH
        else:
            dr = dc = 0.0

        # safety + reachable areas per (UP,DOWN,LEFT,RIGHT) = (dr,dc) in row/col space
        directions = [(0,-1),(0,1),(-1,0),(1,0)]  # (drow, dcol)
        safety_scores, reachable_areas = [], []
        for drow, dcol in directions:
            safe_steps, nr, nc = 0, head_r, head_c
            for _ in range(5):
                nr += drow; nc += dcol
                if is_safe(nr, nc, board): safe_steps += 1
                else: break
            safety_scores.append(safe_steps/5.0)
            fr, fc = head_r+drow, head_c+dcol
            area = flood_fill_size(fr, fc, board, limit=50) if is_safe(fr, fc, board) else 0
            reachable_areas.append(area/50.0)

        # quadrant free
        quadrant_free = [0,0,0,0]
        for r in range(GRID_HEIGHT):
            for c in range(GRID_WIDTH):
                if board[r][c] == 0:
                    idx = (0 if c < GRID_WIDTH//2 else 1) + (0 if r < GRID_HEIGHT//2 else 2)
                    quadrant_free[idx] += 1
        total_cells = GRID_WIDTH*GRID_HEIGHT
        quadrant_free = [q/(total_cells/4) for q in quadrant_free]

        # distances to lethal borders
        border_distances = [
            head_r / GRID_HEIGHT,
            (GRID_HEIGHT-1 - head_r) / GRID_HEIGHT,
            head_c / GRID_WIDTH,
            (GRID_WIDTH-1 - head_c) / GRID_WIDTH,
        ]

        # opponent distance
        if other_trail:
            ox, oy = other_trail[-1]
            opp_dist = manhattan_distance(head_r, head_c, oy, ox) / (GRID_WIDTH+GRID_HEIGHT)
        else:
            opp_dist = 1.0

        open_neighbors = count_open_neighbors(head_r, head_c, board) / 4.0

        # local wall density
        wall_density = []
        for radius in [1,2,3]:
            walls = cells = 0
            for rr in range(-radius, radius+1):
                for cc in range(-radius, radius+1):
                    r0, c0 = head_r+rr, head_c+cc
                    if in_bounds(r0, c0):
                        cells += 1
                        if board[r0][c0] != 0: walls += 1
            wall_density.append(walls/cells if cells else 0.0)

        my_len = len(my_trail)/total_cells
        other_len = len(other_trail)/total_cells
        turn_count = state.get("turn_count", 0)/500.0
        free_cells = sum(1 for r in range(GRID_HEIGHT) for c in range(GRID_WIDTH) if board[r][c]==0)/total_cells

        feats = []
        feats += [norm_r, norm_c, dr, dc]
        feats += safety_scores
        feats += reachable_areas
        feats += quadrant_free
        feats += border_distances
        feats += [my_boosts/3.0, other_boosts/3.0, opp_dist, open_neighbors]
        feats += wall_density
        feats += [my_len, other_len]
        feats.append(turn_count)
        feats.append(free_cells)
        while len(feats) < 128: feats.append(0.0)
        return np.array(feats[:128], dtype=np.float32)
    
    def get_valid_actions(self, state, player_num):
        """Get list of valid actions (not opposite to current direction and in bounds)."""
        board = state.get("board", [[0]*GRID_WIDTH for _ in range(GRID_HEIGHT)])
        
        if player_num == 1:
            my_trail = state.get("agent1_trail", [])
        else:
            my_trail = state.get("agent2_trail", [])
        
        if len(my_trail) < 2:
            # At start, check which directions are actually safe
            if len(my_trail) == 1:
                head_x, head_y = my_trail[0]
                head_r, head_c = head_y, head_x
                valid = []
                directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # UP, DOWN, LEFT, RIGHT
                for idx, (dr, dc) in enumerate(directions):
                    nr, nc = head_r + dr, head_c + dc
                    if is_safe(nr, nc, board, GRID_HEIGHT, GRID_WIDTH):
                        valid.append(idx)
                return valid if valid else [0, 1, 2, 3]
            return [0, 1, 2, 3]
        
        # Calculate current direction
        head = my_trail[-1]
        prev = my_trail[-2]
        
        head_x, head_y = head
        prev_x, prev_y = prev
        head_r, head_c = head_y, head_x
        prev_r, prev_c = prev_y, prev_x
        
        dr = head_r - prev_r
        dc = head_c - prev_c
        
        # Determine current direction
        current_action = None
        if dr == -1 and dc == 0:
            current_action = 0  # UP
        elif dr == 1 and dc == 0:
            current_action = 1  # DOWN
        elif dr == 0 and dc == -1:
            current_action = 2  # LEFT
        elif dr == 0 and dc == 1:
            current_action = 3  # RIGHT
        
        # Return all actions except opposite
        valid = []
        opposite = {0: 1, 1: 0, 2: 3, 3: 2}
        
        # Check each direction for safety
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # UP, DOWN, LEFT, RIGHT
        for idx, (check_dr, check_dc) in enumerate(directions):
            # Skip opposite direction
            if current_action is not None and idx == opposite.get(current_action):
                continue
            
            nr, nc = head_r + check_dr, head_c + check_dc
            # Include even if not immediately safe - model will learn to avoid
            valid.append(idx)
        
        return valid if valid else [0, 1, 2, 3]
    
    def choose_action(self, state, player_num, use_boost_strategy=True):
        """Choose action using the model."""
        features = self.extract_features(state, player_num)
        valid_actions = self.get_valid_actions(state, player_num)
        
        # Get Q-values
        with torch.no_grad():
            state_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
            q_values = self.model(state_tensor).cpu().numpy()[0]
        
        # Mask invalid actions
        masked_q = np.full(4, -float('inf'))
        for action in valid_actions:
            masked_q[action] = q_values[action]
        
        # Choose best valid action
        action = np.argmax(masked_q)
        action_str = self.action_str_map[action]
        
        # Boost strategy
        use_boost = False
        if use_boost_strategy:
            board = state.get("board", [[0]*GRID_WIDTH for _ in range(GRID_HEIGHT)])
            if player_num == 1:
                boosts = state.get("agent1_boosts", 0)
                trail = state.get("agent1_trail", [])
            else:
                boosts = state.get("agent2_boosts", 0)
                trail = state.get("agent2_trail", [])
            
            turn_count = state.get("turn_count", 0)
            
            if boosts > 0 and trail:
                head_x, head_y = trail[-1]
                head_r, head_c = head_y, head_x
                
                # Use boost when:
                # 1. Mid-game (40-120 turns)
                # 2. Close to borders
                # 3. Surrounded by walls
                
                # Check proximity to lethal borders
                near_border = (head_r < 3 or head_r >= GRID_HEIGHT - 3 or 
                              head_c < 3 or head_c >= GRID_WIDTH - 3)
                
                # Count walls near head
                walls_near = 0
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        check_r = head_r + dr
                        check_c = head_c + dc
                        if not in_bounds(check_r, check_c, GRID_HEIGHT, GRID_WIDTH) or board[check_r][check_c] != 0:
                            walls_near += 1
                
                # Use boost if in danger
                if 40 <= turn_count <= 120 and (walls_near >= 6 or near_border):
                    use_boost = True
        
        if use_boost:
            return f"{action_str}:BOOST"
        return action_str

agent = TronAgent()

def _update_local_game_from_post(data: dict):
    """Update local game state."""
    with game_lock:
        LAST_POSTED_STATE.clear()
        LAST_POSTED_STATE.update(data)

        if "board" in data:
            try:
                GLOBAL_GAME.board.grid = data["board"]
            except Exception:
                pass

        if "agent1_trail" in data:
            GLOBAL_GAME.agent1.trail = deque(tuple(p) for p in data["agent1_trail"])
        if "agent2_trail" in data:
            GLOBAL_GAME.agent2.trail = deque(tuple(p) for p in data["agent2_trail"])

        if "agent1_length" in data:
            GLOBAL_GAME.agent1.length = int(data["agent1_length"])
        if "agent2_length" in data:
            GLOBAL_GAME.agent2.length = int(data["agent2_length"])

        if "agent1_alive" in data:
            GLOBAL_GAME.agent1.alive = bool(data["agent1_alive"])
        if "agent2_alive" in data:
            GLOBAL_GAME.agent2.alive = bool(data["agent2_alive"])

        if "agent1_boosts" in data:
            GLOBAL_GAME.agent1.boosts_remaining = int(data["agent1_boosts"])
        if "agent2_boosts" in data:
            GLOBAL_GAME.agent2.boosts_remaining = int(data["agent2_boosts"])

        if "turn_count" in data:
            GLOBAL_GAME.turns = int(data["turn_count"])


@app.route('/')
def index():
    return jsonify({"participant": PARTICIPANT, "agent_name": AGENT_NAME})

# Original handlers
@app.post("/send-state")
def send_state():
    data = request.get_json()
    if not data:
        return jsonify({"error":"no json body"}), 400
    _update_local_game_from_post(data)
    return jsonify({"status":"state received"})

@app.get("/send-move")
def send_move():
    player_number = request.args.get("player_number", default=1, type=int)
    with game_lock:
        state = dict(LAST_POSTED_STATE)
    move = agent.choose_action(state, player_number)
    return jsonify({"move": move})

@app.post("/end")
def end():
    data = request.get_json(silent=True) or {}
    _update_local_game_from_post(data)
    return jsonify({"status":"acknowledged"})

# ----- Judge-friendly aliases (AFTER handlers exist) -----
@app.get("/health")
def health_json(): return jsonify({"ok": True})

@app.get("/ready")
def ready(): return "ready"

@app.post("/state")
def state_alias(): return send_state()

@app.get("/move")
def move_get(): return send_move()

@app.post("/move")
def move_post(): return send_move()

@app.post("/endgame")
def end_alias(): return end()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5008"))
    app.run(host="0.0.0.0", port=port, debug=False)
