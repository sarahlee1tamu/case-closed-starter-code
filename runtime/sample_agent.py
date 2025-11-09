# runtime/sample_agent.py
import os
import glob
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from flask import Flask, request, jsonify

from case_closed_game import Game  # Direction/GameResult not needed at runtime

GRID_WIDTH = 20
GRID_HEIGHT = 18
device = torch.device("cpu")
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

def find_latest_model():
    # First, try to find models in the training directory (for local testing)
    search_path = os.path.join("../training", 'model*.pth')
    files = glob.glob(search_path)
    
    if files:
        # Found models in ../training, use the latest one
        return max(files, key=os.path.getmtime)
    
    # If not found, fall back to model.pth in the current directory (for Docker runtime)
    if os.path.exists('model_20251108_184615.pth'):
        return 'model_20251108_184615.pth'
        
    return None

# ----------------- Model -----------------
class DQN(nn.Module):
    def __init__(self, state_size=128, action_size=4):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, action_size)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

# ----------------- Helper -----------------
class AgentHelper:
    def __init__(self):
        self.policy_net = DQN().to(device)
        latest_model_path = find_latest_model()

        if latest_model_path:
            print(f"Loading latest model from {latest_model_path}")
            checkpoint = torch.load(latest_model_path, map_location="cpu")
            if isinstance(checkpoint, dict) and 'policy_state_dict' in checkpoint:
                self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
                print("Loaded full checkpoint (policy_state_dict).")
            else:
                self.policy_net.load_state_dict(checkpoint)
                print("Loaded old model format (state_dict only).")
        else:
            print("No model found. Using untrained model.")
        
        self.policy_net.eval()
        self.action_map = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}

    def in_bounds(self, r, c): return 0 <= r < GRID_HEIGHT and 0 <= c < GRID_WIDTH
    def is_safe(self, r, c, board): return self.in_bounds(r, c) and board[r][c] == 0

    def flood_fill(self, sr, sc, board, limit=50):
        if not self.is_safe(sr, sc, board): return 0
        seen = [[False]*GRID_WIDTH for _ in range(GRID_HEIGHT)]
        q = deque([(sr, sc)])
        seen[sr][sc] = True
        dirs = [(0,-1),(0,1),(-1,0),(1,0)]
        size = 0
        while q and size < limit:
            r, c = q.popleft(); size += 1
            for dr, dc in dirs:
                nr, nc = r+dr, c+dc
                if self.in_bounds(nr, nc) and not seen[nr][nc] and board[nr][nc] == 0:
                    seen[nr][nc] = True; q.append((nr, nc))
        return size

    def extract_features(self, game, player_num):
        board = game.board.grid
        me  = game.agent1 if player_num == 1 else game.agent2
        opp = game.agent2 if player_num == 1 else game.agent1

        if not me.alive or len(me.trail) == 0:
            return np.zeros(128, dtype=np.float32)

        head_x, head_y = me.trail[-1]
        head_c, head_r = head_x, head_y

        norm_r = head_r / GRID_HEIGHT
        norm_c = head_c / GRID_WIDTH

        if len(me.trail) >= 2:
            prev_x, prev_y = me.trail[-2]
            dr = (head_y - prev_y) / GRID_HEIGHT
            dc = (head_x - prev_x) / GRID_WIDTH
        else:
            dr = dc = 0.0

        dirs = [(0,-1),(0,1),(-1,0),(1,0)]  # UP,DOWN,LEFT,RIGHT in (drow,dcol)
        safety_scores, reachable_areas = [], []
        for drow, dcol in dirs:
            safe_steps, nr, nc = 0, head_r, head_c
            for _ in range(5):
                nr += drow; nc += dcol
                if self.is_safe(nr, nc, board): safe_steps += 1
                else: break
            safety_scores.append(safe_steps / 5.0)
            fr, fc = head_r + drow, head_c + dcol
            area = self.flood_fill(fr, fc, board) if self.is_safe(fr, fc, board) else 0
            reachable_areas.append(area / 50.0)

        quadrant_free = [0,0,0,0]
        for r in range(GRID_HEIGHT):
            for c in range(GRID_WIDTH):
                if board[r][c] == 0:
                    idx = (0 if c < GRID_WIDTH//2 else 1) + (0 if r < GRID_HEIGHT//2 else 2)
                    quadrant_free[idx] += 1
        total_cells = GRID_WIDTH * GRID_HEIGHT
        quadrant_free = [q/(total_cells/4) for q in quadrant_free]

        dist_top    = head_r / GRID_HEIGHT
        dist_bottom = (GRID_HEIGHT-1 - head_r) / GRID_HEIGHT
        dist_left   = head_c / GRID_WIDTH
        dist_right  = (GRID_WIDTH-1 - head_c) / GRID_WIDTH
        border_dists = [dist_top, dist_bottom, dist_left, dist_right]

        if opp.alive and len(opp.trail) > 0:
            ox, oy = opp.trail[-1]
            opp_dist = (abs(head_r - oy) + abs(head_c - ox)) / (GRID_WIDTH + GRID_HEIGHT)
        else:
            opp_dist = 1.0

        open_neighbors = 0
        for drow, dcol in dirs:
            rr, cc = head_r + drow, head_c + dcol
            if self.in_bounds(rr, cc) and board[rr][cc] == 0: open_neighbors += 1
        open_neighbors /= 4.0

        wall_density = []
        for radius in [1,2,3]:
            walls = cells = 0
            for rr in range(-radius, radius+1):
                for cc in range(-radius, radius+1):
                    r0, c0 = head_r+rr, head_c+cc
                    if self.in_bounds(r0, c0):
                        cells += 1
                        if board[r0][c0] != 0: walls += 1
            wall_density.append(walls/cells if cells else 0.0)

        feats = []
        feats += [norm_r, norm_c, dr, dc]
        feats += safety_scores
        feats += reachable_areas
        feats += quadrant_free
        feats += border_dists
        feats += [me.boosts_remaining/3.0, opp.boosts_remaining/3.0, opp_dist, open_neighbors]
        feats += wall_density
        feats += [me.length/total_cells, opp.length/total_cells]
        feats.append(game.turns / 500.0)
        free_cells = sum(1 for r in range(GRID_HEIGHT) for c in range(GRID_WIDTH) if board[r][c] == 0)
        feats.append(free_cells / total_cells)
        while len(feats) < 128: feats.append(0.0)
        return np.array(feats[:128], dtype=np.float32)

    def valid_actions(self, agent):
        if len(agent.trail) < 2:
            return [0,1,2,3]
        (hx, hy), (px, py) = agent.trail[-1], agent.trail[-2]
        dr, dc = hy - py, hx - px
        cur = None
        if   dr == -1 and dc == 0: cur = 0
        elif dr ==  1 and dc == 0: cur = 1
        elif dr ==  0 and dc == -1: cur = 2
        elif dr ==  0 and dc ==  1: cur = 3
        valid = [0,1,2,3]
        if cur is not None:
            opp = {0:1,1:0,2:3,3:2}[cur]
            if opp in valid: valid.remove(opp)
        return valid

# --------------- Flask app & game state ---------------
app = Flask(__name__)
agent = AgentHelper()
game = Game()

@app.route('/')
def index():
    return jsonify({"participant": "SampleParticipant", "agent_name": "SampleAgent"})

@app.post("/send-state")
def send_state():
    data = request.get_json(force=True)
    game.board.grid = data["board"]
    
    # Agent 1
    game.agent1.trail = deque(map(tuple, data["agent1_trail"]))
    game.agent1.alive = data["agent1_alive"]
    game.agent1.length = data["agent1_length"]
    game.agent1.boosts_remaining = data["agent1_boosts"]
    
    # Agent 2
    game.agent2.trail = deque(map(tuple, data["agent2_trail"]))
    game.agent2.alive = data["agent2_alive"]
    game.agent2.length = data["agent2_length"]
    game.agent2.boosts_remaining = data["agent2_boosts"]
    
    game.turns = data["turn_count"]
    return jsonify({"ok": True})

@app.get("/send-move")
def send_move():
    state = agent.extract_features(game, player_num=2)
    valid = agent.valid_actions(game.agent2)
    with torch.inference_mode():
        q = agent.policy_net(torch.from_numpy(state).float().unsqueeze(0).to(device))[0].cpu().numpy()
    masked = np.full(4, -np.inf)
    for a in valid: masked[a] = q[a]
    action = int(np.argmax(masked))
    move = agent.action_map[action]
    return jsonify({"move": move})

@app.post("/end")
def end():
    data = request.get_json(silent=True) or {}
    print("Game over:", data.get("result"))
    return jsonify({"ok": True})

# -------- judge-friendly aliases (after handlers exist) --------
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
    app.run(host="0.0.0.0", port=5009)
