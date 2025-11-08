# training/train.py
import os
import sys
import math
import time
import random
import argparse
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Optional visualization (safe if not installed when --no-viz)
try:
    import pygame
    HAS_PYGAME = True
except Exception:
    HAS_PYGAME = False

# ---- Game API ----
# Assumes you have the same file in training/ as in runtime/
# with classes: Game, Direction, GameResult
from case_closed_game import Game, Direction, GameResult

# -------------------
# Constants / Config
# -------------------
GRID_WIDTH  = 20
GRID_HEIGHT = 18
STATE_SIZE  = 128
ACTION_SIZE = 4  # UP, DOWN, LEFT, RIGHT

DEVICE = torch.device("cpu")
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# -------------
# DQN network
# -------------
class DQN(nn.Module):
    def __init__(self, state_size=STATE_SIZE, action_size=ACTION_SIZE):
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

# -----------------
# Replay Buffer
# -----------------
class ReplayBuffer:
    def __init__(self, capacity=100_000):
        self.buf = deque(maxlen=capacity)
    def __len__(self): return len(self.buf)
    def push(self, s, a, r, ns, d): self.buf.append((s, a, r, ns, d))
    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (np.array(s), np.array(a), np.array(r, dtype=np.float32),
                np.array(ns), np.array(d, dtype=np.float32))

# -----------------
# Feature Extractor
# -----------------
def in_bounds(r, c):
    return 0 <= r < GRID_HEIGHT and 0 <= c < GRID_WIDTH

def is_safe(r, c, board):
    # **Borders are lethal** (no wrap). Out of bounds -> unsafe.
    return in_bounds(r, c) and (board[r][c] == 0)

def flood_fill(sr, sc, board, limit=50):
    if not is_safe(sr, sc, board):
        return 0
    seen = [[False]*GRID_WIDTH for _ in range(GRID_HEIGHT)]
    q = deque([(sr, sc)])
    seen[sr][sc] = True
    size = 0
    dirs = [(0,-1),(0,1),(-1,0),(1,0)]
    while q and size < limit:
        r, c = q.popleft()
        size += 1
        for dr, dc in dirs:
            nr, nc = r+dr, c+dc
            if in_bounds(nr, nc) and not seen[nr][nc] and board[nr][nc] == 0:
                seen[nr][nc] = True
                q.append((nr, nc))
    return size

def extract_features(game, player_num):
    """
    128-dim state with border awareness, local safety, reachable area, etc.
    """
    board = game.board.grid
    me  = game.agent1 if player_num == 1 else game.agent2
    opp = game.agent2 if player_num == 1 else game.agent1

    if not me.alive or len(me.trail) == 0:
        return np.zeros(STATE_SIZE, dtype=np.float32)

    head_x, head_y = me.trail[-1]  # (x=col, y=row)
    head_c, head_r = head_x, head_y

    norm_r = head_r / GRID_HEIGHT
    norm_c = head_c / GRID_WIDTH

    if len(me.trail) >= 2:
        prev_x, prev_y = me.trail[-2]
        dr = (head_y - prev_y) / GRID_HEIGHT
        dc = (head_x - prev_x) / GRID_WIDTH
    else:
        dr, dc = 0.0, 0.0

    dirs = [(0,-1),(0,1),(-1,0),(1,0)]  # UP, DOWN, LEFT, RIGHT
    safety_scores, reachable_areas = [], []
    for dir_r, dir_c in dirs:
        # how many safe steps ahead (max 5)
        safe_steps, nr, nc = 0, head_r, head_c
        for _ in range(5):
            nr += dir_r; nc += dir_c
            if is_safe(nr, nc, board): safe_steps += 1
            else: break
        safety_scores.append(safe_steps/5.0)

        fr, fc = head_r+dir_r, head_c+dir_c
        area = flood_fill(fr, fc, board) if is_safe(fr, fc, board) else 0
        reachable_areas.append(area/50.0)

    # Quadrant free
    quadrant_free = [0,0,0,0]
    for r in range(GRID_HEIGHT):
        for c in range(GRID_WIDTH):
            if board[r][c] == 0:
                idx = (0 if c < GRID_WIDTH//2 else 1) + (0 if r < GRID_HEIGHT//2 else 2)
                quadrant_free[idx] += 1
    total_cells = GRID_WIDTH * GRID_HEIGHT
    quadrant_free = [q/(total_cells/4) for q in quadrant_free]

    # Distances to lethal borders
    dist_top    = head_r / GRID_HEIGHT
    dist_bottom = (GRID_HEIGHT-1 - head_r) / GRID_HEIGHT
    dist_left   = head_c / GRID_WIDTH
    dist_right  = (GRID_WIDTH-1 - head_c) / GRID_WIDTH
    border_dists = [dist_top, dist_bottom, dist_left, dist_right]

    # Opponent distance (Manhattan, normalized)
    if opp.alive and len(opp.trail) > 0:
        ox, oy = opp.trail[-1]
        opp_dist = (abs(head_r - oy) + abs(head_c - ox)) / (GRID_WIDTH + GRID_HEIGHT)
    else:
        opp_dist = 1.0

    # Open neighbors
    open_neighbors = 0
    for dr2, dc2 in dirs:
        rr, cc = head_r+dr2, head_c+dc2
        if in_bounds(rr, cc) and board[rr][cc] == 0:
            open_neighbors += 1
    open_neighbors /= 4.0

    # Wall density in local windows
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

    feats = []
    feats += [norm_r, norm_c, dr, dc]
    feats += safety_scores
    feats += reachable_areas
    feats += quadrant_free
    feats += border_dists
    feats += [me.boosts_remaining/3.0, opp.boosts_remaining/3.0, opp_dist, open_neighbors]
    feats += wall_density
    feats += [me.length/total_cells, opp.length/total_cells]
    feats.append(game.turns/500.0)
    free_cells = sum(1 for r in range(GRID_HEIGHT) for c in range(GRID_WIDTH) if board[r][c] == 0)
    feats.append(free_cells/total_cells)

    while len(feats) < STATE_SIZE:
        feats.append(0.0)

    return np.array(feats[:STATE_SIZE], dtype=np.float32)

# -----------------
# Action helpers
# -----------------
ACTION_TO_DIR = {
    0: Direction.UP,
    1: Direction.DOWN,
    2: Direction.LEFT,
    3: Direction.RIGHT,
}

def valid_actions_for(agent):
    """
    Forbids reversing into the previous cell (classic snake/tron rule).
    Note: the game engine still enforces walls/borders—this just prunes obviously bad actions.
    """
    if len(agent.trail) < 2:
        return [0,1,2,3]
    (hx, hy), (px, py) = agent.trail[-1], agent.trail[-2]
    dr, dc = hy - py, hx - px
    cur = None
    if   dr == -1 and dc == 0: cur = 0  # up
    elif dr ==  1 and dc == 0: cur = 1  # down
    elif dr ==  0 and dc == -1: cur = 2  # left
    elif dr ==  0 and dc ==  1: cur = 3  # right
    valid = [0,1,2,3]
    if cur is not None:
        opp = {0:1,1:0,2:3,3:2}[cur]
        if opp in valid:
            valid.remove(opp)
    return valid

# -----------------
# Opponent policy
# -----------------
def simple_opponent_action(game):
    """
    Border-aware heuristic opponent for training.
    Chooses the valid move with the most free steps ahead (ties random).
    """
    agent = game.agent2
    board = game.board.grid
    cand = valid_actions_for(agent)
    if len(agent.trail) == 0 or not cand:
        return random.choice([0,1,2,3])
    hx, hy = agent.trail[-1]
    head_r, head_c = hy, hx
    dirs = [(0,-1),(0,1),(-1,0),(1,0)]
    best, best_score = cand[0], -1e9
    for a in cand:
        dr, dc = dirs[a]
        nr, nc = head_r+dr, head_c+dc
        if not in_bounds(nr, nc) or board[nr][nc] != 0:
            continue
        score, rr, cc = 0, nr, nc
        for _ in range(5):
            if in_bounds(rr, cc) and board[rr][cc] == 0:
                score += 1
            rr += dr; cc += dc
        # preference to stay off borders a bit
        if nr < 2 or nr >= GRID_HEIGHT-2: score -= 2
        if nc < 2 or nc >= GRID_WIDTH-2:  score -= 2
        if score > best_score:
            best_score, best = score, a
    return best

# -----------------
# Trainer
# -----------------
class Trainer:
    def __init__(self, args):
        self.gamma = 0.99
        self.batch_size = args.batch_size
        self.target_update = args.target_update
        self.lr = args.lr
        self.buffer = ReplayBuffer(args.replay)
        self.eps = args.eps_start
        self.eps_min = args.eps_min
        self.eps_decay = args.eps_decay

        self.policy = DQN().to(DEVICE)
        self.target = DQN().to(DEVICE)
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()

        self.opt = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

        self.visualize = args.viz and HAS_PYGAME
        if self.visualize:
            pygame.init()
            self.cell = 30
            self.window = (GRID_WIDTH*self.cell + 300, GRID_HEIGHT*self.cell)
            self.screen = pygame.display.set_mode(self.window)
            pygame.display.set_caption("Case Closed DQN Training")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)
            self.fps = 10

        self.ep_rewards = []
        self.win_flags = []  # 1 if win, else 0

    def _select_action(self, state, valid_actions, train=True):
        if train and random.random() < self.eps:
            return random.choice(valid_actions)
        with torch.inference_mode():
            qs = self.policy(torch.from_numpy(state).float().unsqueeze(0).to(DEVICE))[0].cpu().numpy()
        masked = np.full(ACTION_SIZE, -np.inf)
        for a in valid_actions:
            masked[a] = qs[a]
        return int(np.argmax(masked))

    def _train_step(self):
        if len(self.buffer) < self.batch_size:
            return
        s, a, r, ns, d = self.buffer.sample(self.batch_size)
        s  = torch.as_tensor(s, dtype=torch.float32, device=DEVICE)
        a  = torch.as_tensor(a, dtype=torch.int64, device=DEVICE)
        r  = torch.as_tensor(r, dtype=torch.float32, device=DEVICE)
        ns = torch.as_tensor(ns, dtype=torch.float32, device=DEVICE)
        d  = torch.as_tensor(d, dtype=torch.float32, device=DEVICE)

        q = self.policy(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            nq = self.target(ns).max(1)[0]
            target = r + (1.0 - d) * self.gamma * nq
        loss = self.loss_fn(q, target)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.opt.step()

    def _render(self, game, episode, total_reward):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit(0)
        self.screen.fill((0,0,0))
        # draw grid & trails
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                rect = pygame.Rect(x*self.cell, y*self.cell, self.cell, self.cell)
                val = game.board.grid[y][x]
                if val == 0:
                    pygame.draw.rect(self.screen, (60,60,60), rect, 1)
                else:
                    color = (255,0,0) if (x,y) in game.agent1.trail else (0,0,255)
                    pygame.draw.rect(self.screen, color, rect)
        # heads
        if game.agent1.alive and game.agent1.trail:
            hx, hy = game.agent1.trail[-1]
            pygame.draw.circle(self.screen, (255,255,0),
                               (hx*self.cell + self.cell//2, hy*self.cell + self.cell//2),
                               self.cell//3)
        if game.agent2.alive and game.agent2.trail:
            hx, hy = game.agent2.trail[-1]
            pygame.draw.circle(self.screen, (0,255,0),
                               (hx*self.cell + self.cell//2, hy*self.cell + self.cell//2),
                               self.cell//3)
        # sidebar stats
        stats_x = GRID_WIDTH*self.cell + 10
        lines = [
            f"Episode: {episode}",
            f"Turn: {game.turns}",
            f"Epsilon: {self.eps:.3f}",
            f"Reward: {total_reward:.1f}",
        ]
        if self.ep_rewards:
            avg = np.mean(self.ep_rewards[-100:])
            lines.append(f"Avg Reward(100): {avg:.1f}")
        if self.win_flags:
            win = np.mean(self.win_flags[-100:]) if len(self.win_flags)>=1 else 0.0
            lines.append(f"Win Rate(100): {win:.1%}")
        for i, txt in enumerate(lines):
            surf = self.font.render(txt, True, (255,255,255))
            self.screen.blit(surf, (stats_x, 10 + i*24))
        pygame.display.flip()
        self.clock.tick(self.fps)

    def train(self, episodes, save_path):
        print(f"Training for {episodes} episodes on CPU…")
        for ep in range(1, episodes+1):
            game = Game()
            total_reward = 0.0
            done = False
            result = None

            while not done:
                # Agent 1 (learning)
                s1 = extract_features(game, 1)
                va1 = valid_actions_for(game.agent1)
                a1  = self._select_action(s1, va1, train=True)

                # Agent 2 (opponent heuristic)
                a2  = simple_opponent_action(game)

                dir1 = ACTION_TO_DIR[a1]
                dir2 = ACTION_TO_DIR[a2]
                result = game.step(dir1, dir2)

                # Reward shaping
                if result == GameResult.AGENT1_WIN:
                    r = 100.0; done = True
                elif result == GameResult.AGENT2_WIN:
                    r = -100.0; done = True
                elif result == GameResult.DRAW:
                    r = 0.0; done = True
                else:
                    # Base survival reward
                    r = 1.0

                    # Encourage staying alive longer
                    r += 0.05 * (game.agent1.length - game.agent2.length)

                    # Encourage controlling space
                    my_head = game.agent1.trail[-1]
                    opp_head = game.agent2.trail[-1] if game.agent2.trail else (0, 0)
                    board = game.board.grid
                    my_area = flood_fill(my_head[1], my_head[0], board)
                    opp_area = flood_fill(opp_head[1], opp_head[0], board)
                    r += 0.001 * (my_area - opp_area)

                    # Penalize being near walls
                    hr, hc = my_head[1], my_head[0]
                    min_border = min(hr, GRID_HEIGHT - 1 - hr, hc, GRID_WIDTH - 1 - hc)
                    if min_border < 2:
                        r -= 0.2

                    # Penalize having few open moves
                    open_moves = sum(is_safe(hr+dr, hc+dc, board) for dr, dc in [(0,-1),(0,1),(-1,0),(1,0)])
                    r -= (4 - open_moves) * 0.1


                    ns1 = extract_features(game, 1)
                    self.buffer.push(s1, a1, r, ns1, float(done))
                    total_reward += r

                    # Learner update
                    self._train_step()

                    # Optional render
                    if self.visualize and (ep % 10 == 0):
                        self._render(game, ep, total_reward)

            # Episode end
            self.ep_rewards.append(total_reward)
            self.win_flags.append(1 if result == GameResult.AGENT1_WIN else 0)

            # Target net sync
            if ep % self.target_update == 0:
                self.target.load_state_dict(self.policy.state_dict())

            # Epsilon decay
            self.eps = max(self.eps_min, self.eps * self.eps_decay)

            # Logs
            if ep % 10 == 0:
                avg_r = np.mean(self.ep_rewards[-100:])
                wr = np.mean(self.win_flags[-100:]) if self.win_flags else 0.0
                print(f"[Ep {ep:4d}] avgR(100)={avg_r:6.2f} win(100)={wr:5.1%} eps={self.eps:.3f}")

            # Periodic save
            if ep % 100 == 0:
                torch.save(self.policy.state_dict(), save_path)
                print(f"Saved checkpoint -> {save_path}")

        # Final save
        torch.save(self.policy.state_dict(), save_path)
        print(f"Training complete. Model saved -> {save_path}")

        if self.visualize and HAS_PYGAME:
            pygame.quit()

# ---------------
# CLI Entrypoint
# ---------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--replay", type=int, default=100000)
    p.add_argument("--target-update", type=int, default=10)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-min", type=float, default=0.05)
    p.add_argument("--eps-decay", type=float, default=0.995)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-path", type=str, default="model.pth")
    p.add_argument("--viz", action="store_true", help="Enable Pygame visualization")
    args = p.parse_args()

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Safety: if --viz but pygame missing, disable
    if args.viz and not HAS_PYGAME:
        print("Pygame not installed; continuing headless.", file=sys.stderr)
        args.viz = False

    trainer = Trainer(args)
    trainer.train(args.episodes, args.save_path)

if __name__ == "__main__":
    main()
