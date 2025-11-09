#
import os, random, math, time
from collections import deque, namedtuple
from typing import Tuple, List, Set

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from case_closed_game import Game, Direction

DEVICE = torch.device("cpu")
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# =======================
# Config
# =======================
GRID_W, GRID_H = 20, 18
ACTION_SIZE = 4  # UP, RIGHT, DOWN, LEFT (no boost during training for simplicity)
REPLAY_CAPACITY = 100_000
BATCH_SIZE = 256
GAMMA = 0.99
TAU = 0.005                  # soft target update factor
LR = 1e-3
WARMUP_STEPS = 10_000
EPS_START, EPS_END, EPS_DECAY_STEPS = 1.0, 0.05, 200_000
TRAIN_STEPS = 100_000        # CHANGE THIS FOR STEP COUNT
TARGET_HARD_COPY_EVERY = 10_000
PRINT_EVERY = 2_000

# =======================
# Utilities  (INT-BASED)
# =======================
# IMPORTANT: match the engine's enum order exactly!
# IMPORTANT: match engine order: UP, DOWN, RIGHT, LEFT
UP, DOWN, RIGHT, LEFT = 0, 1, 2, 3
ORDERED_DIRS = [UP, DOWN, RIGHT, LEFT]
OPPOSITE_IDX = {UP: DOWN, DOWN: UP, RIGHT: LEFT, LEFT: RIGHT}

def evaluate(policy_net, num_games=20):
    wins = 0
    for _ in range(num_games):
        env = SimpleEnv()
        s = env.reset()
        done = False
        while not done:
            with torch.no_grad():
                q = policy_net(torch.as_tensor(s[None], dtype=torch.float32))
                a = int(q.argmax())
            s, r, done, _ = env.step(a)
        if r > 0:
            wins += 1
    print(f"[eval] Win rate: {wins}/{num_games}")


def idx_to_direction(a: int) -> Direction:
    # match enum: UP, DOWN, RIGHT, LEFT
    return [Direction.UP, Direction.DOWN, Direction.RIGHT, Direction.LEFT][a]

def dir_to_idx(d: Direction) -> int:
    return {
        Direction.UP: UP,
        Direction.DOWN: DOWN,
        Direction.RIGHT: RIGHT,
        Direction.LEFT: LEFT,
    }[d]

# 180° reverse map consistent with that order
OPPOSITE_IDX = {
    UP:    DOWN,
    DOWN:  UP,
    RIGHT: LEFT,
    LEFT:  RIGHT,
}

def safe_non_reverse_idx(desired_idx: int, cur_dir_idx: int) -> int:
    return cur_dir_idx if desired_idx == OPPOSITE_IDX[cur_dir_idx] else desired_idx


def wrap(x: int, size: int) -> int:
    return x % size

def torus_step(x: int, y: int, a: int, W: int, H: int) -> tuple[int,int]:
    if a == UP:    return x, wrap(y-1, H)
    if a == DOWN:  return x, wrap(y+1, H)
    if a == LEFT:  return wrap(x-1, W), y
    return wrap(x+1, W), y  # RIGHT

def grid_to_occset(grid) -> set[tuple[int,int]]:
    H = len(grid); W = len(grid[0]) if H else 0
    occ = set()
    for y in range(H):
        row = grid[y]
        for x, v in enumerate(row):
            if v == 1:  # occupied
                occ.add((x,y))
    return occ

def run_length(head_xy: tuple[int,int], a: int, occ: set[tuple[int,int]], W: int, H: int, max_steps: int=512) -> int:
    x, y = head_xy
    cnt = 0
    for _ in range(max_steps):
        x, y = torus_step(x, y, a, W, H)
        if (x,y) in occ: break
        cnt += 1
    return cnt

from collections import deque as _deque
def reachable_size(start: tuple[int,int], occ: set[tuple[int,int]], W: int, H: int, limit: int=400) -> int:
    if start in occ: return 0
    q = _deque([start])
    seen = {start}
    count = 0
    while q and count < limit:
        x,y = q.popleft()
        count += 1
        for a in ORDERED_DIRS:
            nx, ny = torus_step(x, y, a, W, H)
            if (nx,ny) not in occ and (nx,ny) not in seen:
                seen.add((nx,ny))
                q.append((nx,ny))
    return count

def best_safe_dir_index(head_xy: tuple[int,int], cur_dir_idx: int, occ_set: set[tuple[int,int]], W: int, H: int) -> int:
    """Return a non-reverse INT action whose next cell is free. Score by run_len/space/trap."""
    scored = []
    for a in ORDERED_DIRS:
        if a == OPPOSITE_IDX.get(cur_dir_idx):
            continue
        nx, ny = torus_step(head_xy[0], head_xy[1], a, W, H)
        if (nx, ny) in occ_set:
            continue
        rl = run_length(head_xy, a, occ_set, W, H)
        nx2, ny2 = torus_step(nx, ny, a, W, H)
        trap = -5 if (nx2, ny2) in occ_set else 0
        occ1 = set(occ_set); occ1.add((nx, ny))
        space = reachable_size((nx, ny), occ1, W, H)
        scored.append((rl, space, trap, -ORDERED_DIRS.index(a), a))
    if not scored:
        # last resort: if literally nothing but reverse is open, allow reverse
        return OPPOSITE_IDX.get(cur_dir_idx, RIGHT)
    scored.sort(reverse=True)
    return scored[0][-1]

# Opponent that cannot reverse; scores by run_len/space with 2-ply trap penalty
def opponent_move(grid, head_xy: tuple[int,int], cur_dir_idx: int) -> int:
    H = len(grid); W = len(grid[0])
    occ = grid_to_occset(grid)
    scored = []
    for a in ORDERED_DIRS:
        if a == OPPOSITE_IDX.get(cur_dir_idx):  # disallow 180° reverse
            continue
        nx, ny = torus_step(head_xy[0], head_xy[1], a, W, H)
        if (nx,ny) in occ:
            continue
        rl = run_length(head_xy, a, occ, W, H)
        nx2, ny2 = torus_step(nx, ny, a, W, H)
        trap = -5 if (nx2, ny2) in occ else 0
        occ1 = set(occ); occ1.add((nx,ny))
        space = reachable_size((nx,ny), occ1, W, H)
        scored.append((rl, space, trap, -ORDERED_DIRS.index(a), a))
    if not scored:
        return OPPOSITE_IDX.get(cur_dir_idx, RIGHT)
    scored.sort(reverse=True)
    return scored[0][-1]

# =======================
# Env wrapper around Game
# =======================
class SimpleEnv:
    """Fresh Game per episode. Agent1 learns; Agent2 is heuristic.
       Observation: 3 x H x W (occupancy, my head, opp head).
       All moves & safety logic are INT-based.
    """
    def __init__(self):
        self.game = None

    def reset(self):
        self.game = Game()
        return self._make_obs()

    def _make_obs(self):
        board = self.game.board
        grid = board.grid
        H = board.height; W = board.width
        occ = np.array(grid, dtype=np.float32)
        me_head = np.zeros_like(occ)
        op_head = np.zeros_like(occ)
        hx, hy = self.game.agent1.trail[-1]
        me_head[hy, hx] = 1.0
        ox, oy = self.game.agent2.trail[-1]
        op_head[oy, ox] = 1.0
        obs = np.stack([occ, me_head, op_head], axis=0)
        return obs

    def step(self, action_idx: int):
        board = self.game.board
        W, H = board.width, board.height
        grid = board.grid
        occ = grid_to_occset(grid)

        # engine's true directions -> INT
        cur_dir1_idx = dir_to_idx(self.game.agent1.direction)
        cur_dir2_idx = dir_to_idx(self.game.agent2.direction)

        # 1) Learner proposed action (INT); block reverse in INT space
        a1_idx = safe_non_reverse_idx(action_idx, cur_dir1_idx)

        # 2) HARD SAFETY: if next cell is occupied, override to best safe dir
        head1 = self.game.agent1.trail[-1]
        nx, ny = torus_step(head1[0], head1[1], a1_idx, W, H)
        if (nx, ny) in occ:
            a1_idx = best_safe_dir_index(head1, cur_dir1_idx, occ, W, H)
        else:
            # Extra guard: if 2-ply trap, switch to better option when available
            nx2, ny2 = torus_step(nx, ny, a1_idx, W, H)
            if (nx2, ny2) in occ:
                alt = best_safe_dir_index(head1, cur_dir1_idx, occ, W, H)
                if alt != a1_idx:
                    a1_idx = alt

        # 3) Opponent (INT) with no reverse
        head2 = self.game.agent2.trail[-1]
        a2_idx = opponent_move(grid, (head2[0], head2[1]), cur_dir2_idx)

        # 4) FINAL VECTOR-SPACE GUARDS (run LAST, after all overrides)
        cur1_dir = self.game.agent1.direction
        cur2_dir = self.game.agent2.direction

        def apply_final_vector_guard(head_xy, cur_dir, idx):
            cur_dx, cur_dy = cur_dir.value
            req_dir = idx_to_direction(idx)
            req_dx, req_dy = req_dir.value
            if (req_dx, req_dy) == (-cur_dx, -cur_dy):
                # pick a lateral alternative that's not reverse and not blocked
                for cand in ORDERED_DIRS:
                    if cand == idx:
                        continue
                    if cand == OPPOSITE_IDX[dir_to_idx(cur_dir)]:
                        continue
                    cx, cy = torus_step(head_xy[0], head_xy[1], cand, W, H)
                    if (cx, cy) not in occ:
                        return cand
                # if totally stuck, keep current heading (engine won’t reject it)
                return dir_to_idx(cur_dir)
            return idx

        a1_idx = apply_final_vector_guard(self.game.agent1.trail[-1], cur1_dir, a1_idx)
        a2_idx = apply_final_vector_guard(self.game.agent2.trail[-1], cur2_dir, a2_idx)

        # 5) Convert to Directions and step ONCE
        a1 = idx_to_direction(a1_idx)
        a2 = idx_to_direction(a2_idx)
        result = self.game.step(a1, a2, False, False)

        # 6) Terminal/base reward
        done = False
        base_reward = 0.0
        if result is not None:
            done = True
            if str(result) == "GameResult.AGENT1_WIN":
                base_reward = 1.0
            elif str(result) == "GameResult.DRAW":
                base_reward = 0.0
            else:
                base_reward = -1.0

        # 7) Shaping: run_length & reachable space after this move
        grid2 = self.game.board.grid
        occ2 = grid_to_occset(grid2)
        my_head = self.game.agent1.trail[-1]
        rl = run_length(my_head, a1_idx, occ2, W, H)
        nxs, nys = torus_step(my_head[0], my_head[1], a1_idx, W, H)
        space = 0
        if (nxs, nys) not in occ2:
            occ_after = set(occ2); occ_after.add((nxs, nys))
            space = reachable_size((nxs, nys), occ_after, W, H)

        shaped = 0.01
        if done and base_reward < 0: shaped -= 1.0
        if rl <= 2: shaped -= 0.02
        shaped += 0.002 * min(space, 400)  # Encourage open space more


        # Bonus for using a boost near opponent (later if boosts are enabled)
# Not yet implemented — stub if needed
# shaped += 0.2 * used_boost_when_close_to_enemy


        reward = base_reward + shaped
        obs_next = self._make_obs() if not done else np.zeros_like(self._make_obs(), dtype=np.float32)
        return obs_next, reward, done, {}





# =======================
# Replay Buffer
# =======================
Transition = namedtuple("Transition", ("s", "a", "r", "sp", "d"))

class Replay:
    def __init__(self, capacity: int):
        self.buf = deque(maxlen=capacity)
    def __len__(self): return len(self.buf)
    def add(self, *args): self.buf.append(Transition(*args))
    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s  = np.stack([b.s for b in batch], axis=0)
        sp = np.stack([b.sp for b in batch], axis=0)
        a  = np.array([b.a  for b in batch], dtype=np.int64)
        r  = np.array([b.r  for b in batch], dtype=np.float32)
        d  = np.array([b.d  for b in batch], dtype=np.float32)
        return s, a, r, sp, d

# =======================
# Model
# =======================
class DQN(nn.Module):
    def __init__(self, in_channels=3, H=GRID_H, W=GRID_W, action_dim=ACTION_SIZE):
        super().__init__()
        # Small conv net -> MLP head
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        conv_out = 32 * H * W
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, action_dim),
        )

    def forward(self, x):
        x = self.conv(x)
        return self.head(x)

# =======================
# Training
# =======================
def epsilon_at(step: int) -> float:
    frac = min(1.0, max(0.0, step / EPS_DECAY_STEPS))
    return EPS_START + (EPS_END - EPS_START) * frac

def soft_update_(target_net, policy_net, tau=TAU):
    with torch.no_grad():
        for tp, pp in zip(target_net.parameters(), policy_net.parameters()):
            tp.data.mul_(1.0 - tau).add_(tau * pp.data)

def train():
    env = SimpleEnv()

    policy = DQN().to(DEVICE)
    target = DQN().to(DEVICE)
    target.load_state_dict(policy.state_dict())
    target.eval()

    checkpoint_path = "dqn.pt"
    if os.path.exists(checkpoint_path):
        print(f"[train] Loading checkpoint from {checkpoint_path}")
        policy.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
        target.load_state_dict(policy.state_dict())  # Sync target


    opt = optim.Adam(policy.parameters(), lr=LR)
    replay = Replay(REPLAY_CAPACITY)

    global_step = 0
    episode = 0
    ep_returns = deque(maxlen=20)

    s = env.reset()
    while global_step < TRAIN_STEPS:
        # Epsilon-greedy
        eps = epsilon_at(global_step)
        if random.random() < eps:
            a = random.randrange(ACTION_SIZE)
        else:
            with torch.no_grad():
                qs = policy(torch.as_tensor(s[None, ...], dtype=torch.float32, device=DEVICE))
                a = int(qs.argmax(dim=1).item())

        sp, r, done, _ = env.step(a)
        replay.add(s, a, r, sp, float(done))

        s = sp
        global_step += 1

        # Learn
        if len(replay) >= max(BATCH_SIZE, WARMUP_STEPS):
            sb, ab, rb, spb, db = replay.sample(BATCH_SIZE)
            sb_t  = torch.as_tensor(sb, dtype=torch.float32, device=DEVICE)
            ab_t  = torch.as_tensor(ab, dtype=torch.int64,   device=DEVICE).unsqueeze(1)
            rb_t  = torch.as_tensor(rb, dtype=torch.float32, device=DEVICE)
            spb_t = torch.as_tensor(spb, dtype=torch.float32, device=DEVICE)
            db_t  = torch.as_tensor(db, dtype=torch.float32, device=DEVICE)

            # Q(s,a)
            q_all = policy(sb_t)                         # (B, A)
            q_sa  = q_all.gather(1, ab_t).squeeze(1)     # (B,)

            # Double DQN target
            with torch.no_grad():
                ap = policy(spb_t).argmax(dim=1, keepdim=True)   # (B,1)
                q_next = target(spb_t).gather(1, ap).squeeze(1) # (B,)
                y = rb_t + (1.0 - db_t) * GAMMA * q_next         # (B,)

            loss = F.smooth_l1_loss(q_sa, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
            opt.step()

            # soft target update
            soft_update_(target, policy, tau=TAU)

        if done:
            # Episode accounting
            ep_returns.append(r)  # terminal reward proxy
            episode += 1
            s = env.reset()

        # Periodic logs & hard sync
        if global_step % PRINT_EVERY == 0:
            avg_ret = float(np.mean(ep_returns)) if ep_returns else 0.0
            print(f"[step {global_step}] eps={eps:.3f} replay={len(replay)} avg_ret={avg_ret:.3f}")

        if global_step % TARGET_HARD_COPY_EVERY == 0:
            target.load_state_dict(policy.state_dict())

        if global_step % 5000 == 0:
            evaluate(policy)


        if global_step % 5000 == 0:
            torch.save(policy.state_dict(), f"dqn_{global_step}.pt")


    # Save
    save_path = "dqn.pt"
    torch.save(policy.state_dict(), save_path)
    print(f"Saved trained weights to {save_path}")

if __name__ == "__main__":
    train()

