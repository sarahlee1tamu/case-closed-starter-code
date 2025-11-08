import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, jsonify
from threading import Lock
from collections import deque

from case_closed_game import Game, Direction, GameResult

# Flask API server setup
app = Flask(__name__)

GLOBAL_GAME = Game()
LAST_POSTED_STATE = {}
game_lock = Lock()

PARTICIPANT = "RLTeam"
AGENT_NAME = "TronRL"

# Device configuration
device = torch.device("cpu")  # CPU-only as per requirements

# Constants for board analysis
GRID_WIDTH = 20
GRID_HEIGHT = 18

class DQN(nn.Module):
    """Deep Q-Network for the agent."""
    def __init__(self, state_size=128, action_size=4):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

def in_bounds(r: int, c: int, rows: int = GRID_HEIGHT, cols: int = GRID_WIDTH) -> bool:
    """Check if position is within board bounds. Border/out-of-bounds is lethal."""
    return 0 <= r < rows and 0 <= c < cols

def is_safe(r: int, c: int, board, rows: int = GRID_HEIGHT, cols: int = GRID_WIDTH) -> bool:
    """Check if a cell is safe to move to. Border is NOT safe."""
    if not in_bounds(r, c, rows, cols):
        return False
    return board[r][c] == 0

def flood_fill_size(sr: int, sc: int, board, rows: int = GRID_HEIGHT, cols: int = GRID_WIDTH, limit: int = None) -> int:
    """BFS to estimate reachable open cells. Treats out-of-bounds as blocked."""
    if not in_bounds(sr, sc, rows, cols) or not (board[sr][sc] == 0):
        return 0
    
    seen = [[False] * cols for _ in range(rows)]
    q = deque([(sr, sc)])
    seen[sr][sc] = True
    size = 0
    lim = limit or (rows * cols)
    
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # UP, DOWN, LEFT, RIGHT
    
    while q and size < lim:
        r, c = q.popleft()
        size += 1
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if in_bounds(nr, nc, rows, cols) and not seen[nr][nc] and board[nr][nc] == 0:
                seen[nr][nc] = True
                q.append((nr, nc))
    
    return size

def count_open_neighbors(r: int, c: int, board, rows: int = GRID_HEIGHT, cols: int = GRID_WIDTH) -> int:
    """Count safe neighboring cells."""
    count = 0
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if in_bounds(nr, nc, rows, cols) and board[nr][nc] == 0:
            count += 1
    return count

def manhattan_distance(r1: int, c1: int, r2: int, c2: int) -> int:
    """Calculate Manhattan distance between two points."""
    return abs(r1 - r2) + abs(c1 - c2)

class TronAgent:
    """RL Agent for Case Closed game."""
    def __init__(self, model_path='model.pth'):
        self.model = DQN().to(device)
        self.model_path = model_path
        
        # Try to load pretrained model
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=device))
                self.model.eval()
                print(f"Loaded model from {model_path}")
            except Exception as e:
                print(f"Could not load model: {e}")
        
        self.action_map = {0: Direction.UP, 1: Direction.DOWN, 2: Direction.LEFT, 3: Direction.RIGHT}
        self.action_str_map = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
        
    def extract_features(self, state, player_num):
        """Extract features from game state with proper boundary handling."""
        board = state.get("board", [[0]*GRID_WIDTH for _ in range(GRID_HEIGHT)])
        
        if player_num == 1:
            my_trail = state.get("agent1_trail", [])
            my_boosts = state.get("agent1_boosts", 3)
            my_alive = state.get("agent1_alive", True)
            other_trail = state.get("agent2_trail", [])
            other_boosts = state.get("agent2_boosts", 3)
        else:
            my_trail = state.get("agent2_trail", [])
            my_boosts = state.get("agent2_boosts", 3)
            my_alive = state.get("agent2_alive", True)
            other_trail = state.get("agent1_trail", [])
            other_boosts = state.get("agent1_boosts", 3)
        
        if not my_trail or not my_alive:
            return np.zeros(128, dtype=np.float32)
        
        # Game uses (x, y) where x=col, y=row
        # Board[y][x] means board[row][col]
        head_x, head_y = my_trail[-1]
        
        # Normalize positions
        norm_x = head_x / GRID_WIDTH
        norm_y = head_y / GRID_HEIGHT
        
        # Direction vector
        if len(my_trail) >= 2:
            prev_x, prev_y = my_trail[-2]
            dx = (head_x - prev_x) / GRID_WIDTH
            dy = (head_y - prev_y) / GRID_HEIGHT
        else:
            dx, dy = 0, 0
        
        # Check safety in each direction (NO WRAPAROUND - borders are lethal)
        # Directions: 0=UP (dy=-1), 1=DOWN (dy=+1), 2=LEFT (dx=-1), 3=RIGHT (dx=+1)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # (dy, dx)
        safety_scores = []
        reachable_areas = []
        
        for d_idx, (dy_dir, dx_dir) in enumerate(directions):
            # Check multiple steps ahead
            safe_steps = 0
            next_x, next_y = head_x, head_y
            
            for step in range(1, 6):  # Look 5 steps ahead
                next_x = next_x + dx_dir
                next_y = next_y + dy_dir
                
                if not (0 <= next_x < GRID_WIDTH and 0 <= next_y < GRID_HEIGHT):
                    break
                if board[next_y][next_x] != 0:
                    break
                safe_steps += 1
            
            safety_scores.append(safe_steps / 5.0)
            
            # Calculate reachable area from first step in this direction
            first_x = head_x + dx_dir
            first_y = head_y + dy_dir
            if 0 <= first_x < GRID_WIDTH and 0 <= first_y < GRID_HEIGHT and board[first_y][first_x] == 0:
                area = flood_fill_size(first_y, first_x, board, GRID_HEIGHT, GRID_WIDTH, limit=50)
                reachable_areas.append(area / 50.0)
            else:
                reachable_areas.append(0.0)
        
        # Count free space in quadrants
        quadrant_free = [0, 0, 0, 0]
        for r in range(GRID_HEIGHT):
            for c in range(GRID_WIDTH):
                if board[r][c] == 0:
                    quad_idx = (0 if c < GRID_WIDTH // 2 else 1) + (0 if r < GRID_HEIGHT // 2 else 2)
                    quadrant_free[quad_idx] += 1
        
        total_cells = GRID_HEIGHT * GRID_WIDTH
        quadrant_free = [q / (total_cells / 4) for q in quadrant_free]
        
        # Distance to borders (important since borders are lethal)
        dist_to_top = head_r / GRID_HEIGHT
        dist_to_bottom = (GRID_HEIGHT - 1 - head_r) / GRID_HEIGHT
        dist_to_left = head_c / GRID_WIDTH
        dist_to_right = (GRID_WIDTH - 1 - head_c) / GRID_WIDTH
        border_distances = [dist_to_top, dist_to_bottom, dist_to_left, dist_to_right]
        
        # Opponent distance
        if other_trail:
            opp_x, opp_y = other_trail[-1]
            opp_r, opp_c = opp_y, opp_x
            opp_dist = manhattan_distance(head_r, head_c, opp_r, opp_c) / (GRID_HEIGHT + GRID_WIDTH)
        else:
            opp_dist = 1.0
        
        # Open neighbors (degree of freedom)
        open_neighbors = count_open_neighbors(head_r, head_c, board, GRID_HEIGHT, GRID_WIDTH) / 4.0
        
        # Wall density around agent
        wall_density = []
        for radius in [1, 2, 3]:
            walls = 0
            cells = 0
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    check_r = head_r + dr
                    check_c = head_c + dc
                    if in_bounds(check_r, check_c, GRID_HEIGHT, GRID_WIDTH):
                        cells += 1
                        if board[check_r][check_c] != 0:
                            walls += 1
            wall_density.append(walls / cells if cells > 0 else 0)
        
        # Build feature vector
        features = []
        
        # Position features (4)
        features.extend([norm_r, norm_c, dr, dc])
        
        # Safety scores (4)
        features.extend(safety_scores)
        
        # Reachable areas (4)
        features.extend(reachable_areas)
        
        # Quadrant features (4)
        features.extend(quadrant_free)
        
        # Border distances (4) - CRITICAL for non-wraparound awareness
        features.extend(border_distances)
        
        # Agent state (4)
        features.extend([my_boosts / 3.0, other_boosts / 3.0, opp_dist, open_neighbors])
        
        # Wall density (3)
        features.extend(wall_density)
        
        # Trail lengths (2)
        my_len = len(my_trail) / total_cells
        other_len = len(other_trail) / total_cells
        features.extend([my_len, other_len])
        
        # Turn count (1)
        turn_count = state.get("turn_count", 0) / 500.0
        features.append(turn_count)
        
        # Total free cells remaining (1)
        free_cells = sum(1 for r in range(GRID_HEIGHT) for c in range(GRID_WIDTH) if board[r][c] == 0)
        features.append(free_cells / total_cells)
        
        # Pad to 128 dimensions
        while len(features) < 128:
            features.append(0.0)
        
        return np.array(features[:128], dtype=np.float32)
    
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

# Global agent instance
agent = TronAgent()

@app.route("/", methods=["GET"])
def info():
    """Basic health/info endpoint."""
    return jsonify({"participant": PARTICIPANT, "agent_name": AGENT_NAME}), 200

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

@app.route("/send-state", methods=["POST"])
def receive_state():
    """Receive game state from judge."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "no json body"}), 400
    _update_local_game_from_post(data)
    return jsonify({"status": "state received"}), 200

@app.route("/send-move", methods=["GET"])
def send_move():
    """Return agent's move."""
    player_number = request.args.get("player_number", default=1, type=int)
    
    with game_lock:
        state = dict(LAST_POSTED_STATE)
    
    # Use RL agent to choose move
    move = agent.choose_action(state, player_number)
    
    return jsonify({"move": move}), 200

@app.route("/end", methods=["POST"])
def end_game():
    """Acknowledge game end."""
    data = request.get_json()
    if data:
        _update_local_game_from_post(data)
    return jsonify({"status": "acknowledged"}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5008"))
    app.run(host="0.0.0.0", port=port, debug=False)