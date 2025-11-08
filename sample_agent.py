import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import pygame
import sys

from case_closed_game import Game, Direction, GameResult

# Initialize Pygame
pygame.init()

# Constants
CELL_SIZE = 30
GRID_WIDTH = 20
GRID_HEIGHT = 18
WINDOW_WIDTH = GRID_WIDTH * CELL_SIZE + 300  # Extra space for stats
WINDOW_HEIGHT = GRID_HEIGHT * CELL_SIZE
FPS = 10  # Frames per second for visualization

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
GRAY = (128, 128, 128)
YELLOW = (255, 255, 0)

device = torch.device("cpu")

class DQN(nn.Module):
    """Deep Q-Network."""
    def __init__(self, state_size=128, action_size=4):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, action_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)

class ReplayBuffer:
    """Experience replay buffer."""
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)

class TronTrainer:
    """Trainer for the RL agent."""
    def __init__(self, visualize=True):
        self.policy_net = DQN().to(device)
        self.target_net = DQN().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0001)
        self.memory = ReplayBuffer(10000)
        
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.gamma = 0.99
        self.batch_size = 64
        self.target_update = 10
        
        self.action_map = {0: Direction.UP, 1: Direction.DOWN, 2: Direction.LEFT, 3: Direction.RIGHT}
        
        self.visualize = visualize
        if visualize:
            self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
            pygame.display.set_caption("Case Closed RL Training")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)
        
        self.episode_rewards = []
        self.win_rate = []
    
    def in_bounds(self, r, c):
        """Check if position is within board bounds (NO WRAPAROUND)."""
        return 0 <= r < GRID_HEIGHT and 0 <= c < GRID_WIDTH
    
    def is_safe(self, r, c, board):
        """Check if cell is safe (in bounds and empty). Border is lethal."""
        if not self.in_bounds(r, c):
            return False
        return board[r][c] == 0
    
    def flood_fill(self, sr, sc, board, limit=50):
        """BFS to count reachable cells. Out-of-bounds treated as blocked."""
        if not self.is_safe(sr, sc, board):
            return 0
        
        seen = [[False] * GRID_WIDTH for _ in range(GRID_HEIGHT)]
        q = deque([(sr, sc)])
        seen[sr][sc] = True
        size = 0
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        
        while q and size < limit:
            r, c = q.popleft()
            size += 1
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if self.in_bounds(nr, nc) and not seen[nr][nc] and board[nr][nc] == 0:
                    seen[nr][nc] = True
                    q.append((nr, nc))
        return size
    
    def extract_features(self, game, player_num):
        """Extract features from game state with proper boundary handling."""
        board = game.board.grid
        
        if player_num == 1:
            my_agent = game.agent1
            other_agent = game.agent2
        else:
            my_agent = game.agent2
            other_agent = game.agent1
        
        if not my_agent.alive or len(my_agent.trail) == 0:
            return np.zeros(128, dtype=np.float32)
        
        # Game uses (x, y) where x=col, y=row
        # Board[y][x] means board[row][col]
        head_x, head_y = my_agent.trail[-1]
        head_c, head_r = head_x, head_y  # x is column, y is row
        
        # Normalize positions
        norm_r = head_r / GRID_HEIGHT
        norm_c = head_c / GRID_WIDTH
        
        # Direction vector
        if len(my_agent.trail) >= 2:
            prev_x, prev_y = my_agent.trail[-2]
            dr = (head_y - prev_y) / GRID_HEIGHT
            dc = (head_x - prev_x) / GRID_WIDTH
        else:
            dr, dc = 0, 0
        
        # Safety in each direction (NO WRAPAROUND - borders are lethal)
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # UP, DOWN, LEFT, RIGHT
        safety_scores = []
        reachable_areas = []
        
        for dir_r, dir_c in directions:
            safe_steps = 0
            next_r, next_c = head_r, head_c
            
            for step in range(1, 6):
                next_r = next_r + dir_r
                next_c = next_c + dir_c
                
                if self.is_safe(next_r, next_c, board):
                    safe_steps += 1
                else:
                    break
            
            safety_scores.append(safe_steps / 5.0)
            
            # Calculate reachable area
            first_r = head_r + dir_r
            first_c = head_c + dir_c
            if self.is_safe(first_r, first_c, board):
                area = self.flood_fill(first_r, first_c, board)
                reachable_areas.append(area / 50.0)
            else:
                reachable_areas.append(0.0)
        
        # Quadrant free space
        quadrant_free = [0, 0, 0, 0]
        for r in range(GRID_HEIGHT):
            for c in range(GRID_WIDTH):
                if board[r][c] == 0:
                    quad_idx = (0 if c < GRID_WIDTH//2 else 1) + (0 if r < GRID_HEIGHT//2 else 2)
                    quadrant_free[quad_idx] += 1
        
        total_cells = GRID_WIDTH * GRID_HEIGHT
        quadrant_free = [q / (total_cells / 4) for q in quadrant_free]
        
        # Distance to lethal borders (CRITICAL for non-wraparound)
        dist_to_top = head_r / GRID_HEIGHT
        dist_to_bottom = (GRID_HEIGHT - 1 - head_r) / GRID_HEIGHT
        dist_to_left = head_c / GRID_WIDTH
        dist_to_right = (GRID_WIDTH - 1 - head_c) / GRID_WIDTH
        border_distances = [dist_to_top, dist_to_bottom, dist_to_left, dist_to_right]
        
        # Opponent distance
        if other_agent.alive and len(other_agent.trail) > 0:
            opp_x, opp_y = other_agent.trail[-1]
            opp_r, opp_c = opp_y, opp_x
            # Manhattan distance (no wraparound)
            opp_dist = (abs(head_r - opp_r) + abs(head_c - opp_c)) / (GRID_WIDTH + GRID_HEIGHT)
        else:
            opp_dist = 1.0
        
        # Open neighbors
        open_neighbors = 0
        for dr, dc in directions:
            nr, nc = head_r + dr, head_c + dc
            if self.in_bounds(nr, nc) and board[nr][nc] == 0:
                open_neighbors += 1
        open_neighbors = open_neighbors / 4.0
        
        # Wall density
        wall_density = []
        for radius in [1, 2, 3]:
            walls = 0
            cells = 0
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    check_r = head_r + dr
                    check_c = head_c + dc
                    if self.in_bounds(check_r, check_c):
                        cells += 1
                        if board[check_r][check_c] != 0:
                            walls += 1
            wall_density.append(walls / cells if cells > 0 else 0)
        
        # Build feature vector
        features = []
        features.extend([norm_r, norm_c, dr, dc])
        features.extend(safety_scores)
        features.extend(reachable_areas)
        features.extend(quadrant_free)
        features.extend(border_distances)  # NEW: Border awareness
        features.extend([my_agent.boosts_remaining / 3.0, other_agent.boosts_remaining / 3.0, opp_dist, open_neighbors])
        features.extend(wall_density)
        features.extend([my_agent.length / total_cells, other_agent.length / total_cells])
        features.append(game.turns / 500.0)
        
        # Free cells remaining
        free_cells = sum(1 for r in range(GRID_HEIGHT) for c in range(GRID_WIDTH) if board[r][c] == 0)
        features.append(free_cells / total_cells)
        
        # Pad to 128 dimensions
        while len(features) < 128:
            features.append(0.0)
        
        return np.array(features[:128], dtype=np.float32)
    
    def get_valid_actions(self, agent):
        """Get valid actions for an agent (no opposite direction, respect borders)."""
        if len(agent.trail) < 2:
            return [0, 1, 2, 3]
        
        head = agent.trail[-1]
        prev = agent.trail[-2]
        
        head_x, head_y = head
        prev_x, prev_y = prev
        head_r, head_c = head_y, head_x
        prev_r, prev_c = prev_y, prev_x
        
        dr = head_r - prev_r
        dc = head_c - prev_c
        
        current_action = None
        if dr == -1 and dc == 0:
            current_action = 0  # UP
        elif dr == 1 and dc == 0:
            current_action = 1  # DOWN
        elif dr == 0 and dc == -1:
            current_action = 2  # LEFT
        elif dr == 0 and dc == 1:
            current_action = 3  # RIGHT
        
        valid = [0, 1, 2, 3]
        if current_action is not None:
            opposite = {0: 1, 1: 0, 2: 3, 3: 2}
            if opposite[current_action] in valid:
                valid.remove(opposite[current_action])
        
        return valid
    
    def select_action(self, state, valid_actions, training=True):
        """Select action using epsilon-greedy."""
        if training and random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.policy_net(state_tensor).cpu().numpy()[0]
        
        masked_q = np.full(4, -float('inf'))
        for action in valid_actions:
            masked_q[action] = q_values[action]
        
        return np.argmax(masked_q)
    
    def render(self, game, episode, total_reward, epsilon):
        """Render the game state."""
        if not self.visualize:
            return
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
        self.screen.fill(BLACK)
        
        # Draw grid
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                
                if game.board.grid[y][x] == 0:
                    pygame.draw.rect(self.screen, GRAY, rect, 1)
                else:
                    # Check which agent's trail
                    if (x, y) in game.agent1.trail:
                        color = RED if game.agent1.alive else (100, 0, 0)
                        pygame.draw.rect(self.screen, color, rect)
                    elif (x, y) in game.agent2.trail:
                        color = BLUE if game.agent2.alive else (0, 0, 100)
                        pygame.draw.rect(self.screen, color, rect)
        
        # Draw heads
        if game.agent1.alive and len(game.agent1.trail) > 0:
            head = game.agent1.trail[-1]
            pygame.draw.circle(self.screen, YELLOW, 
                             (head[0] * CELL_SIZE + CELL_SIZE // 2, 
                              head[1] * CELL_SIZE + CELL_SIZE // 2), 
                             CELL_SIZE // 3)
        
        if game.agent2.alive and len(game.agent2.trail) > 0:
            head = game.agent2.trail[-1]
            pygame.draw.circle(self.screen, GREEN, 
                             (head[0] * CELL_SIZE + CELL_SIZE // 2, 
                              head[1] * CELL_SIZE + CELL_SIZE // 2), 
                             CELL_SIZE // 3)
        
        # Draw stats
        stats_x = GRID_WIDTH * CELL_SIZE + 10
        stats = [
            f"Episode: {episode}",
            f"Turn: {game.turns}",
            f"Epsilon: {epsilon:.3f}",
            f"Reward: {total_reward:.1f}",
            f"",
            f"Agent 1 (Learning):",
            f"  Alive: {game.agent1.alive}",
            f"  Length: {game.agent1.length}",
            f"  Boosts: {game.agent1.boosts_remaining}",
            f"",
            f"Agent 2 (Opponent):",
            f"  Alive: {game.agent2.alive}",
            f"  Length: {game.agent2.length}",
            f"  Boosts: {game.agent2.boosts_remaining}",
        ]
        
        if len(self.episode_rewards) > 0:
            stats.append(f"")
            stats.append(f"Avg Reward (100): {np.mean(self.episode_rewards[-100:]):.1f}")
        
        if len(self.win_rate) > 0:
            stats.append(f"Win Rate (100): {np.mean(self.win_rate[-100:]):.2%}")
        
        for i, text in enumerate(stats):
            surface = self.font.render(text, True, WHITE)
            self.screen.blit(surface, (stats_x, 10 + i * 25))
        
        pygame.display.flip()
        self.clock.tick(FPS)
    
    def opponent_policy(self, game):
        """Simple opponent policy with border awareness."""
        agent = game.agent2
        valid_actions = self.get_valid_actions(agent)
        board = game.board.grid
        
        if len(agent.trail) == 0:
            return random.choice(valid_actions)
        
        head = agent.trail[-1]
        head_x, head_y = head
        head_r, head_c = head_y, head_x
        
        best_action = valid_actions[0]
        best_score = -1
        
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # UP, DOWN, LEFT, RIGHT
        
        for action in valid_actions:
            direction = self.action_map[action]
            dr, dc = directions[action]
            next_r = head_r + dr
            next_c = head_c + dc
            
            # Penalize moving out of bounds or into walls heavily
            if not self.in_bounds(next_r, next_c):
                continue
            
            if board[next_r][next_c] != 0:
                continue
            
            # Count free cells ahead (no wraparound)
            score = 0
            check_r, check_c = next_r, next_c
            for _ in range(5):
                if self.in_bounds(check_r, check_c) and board[check_r][check_c] == 0:
                    score += 1
                check_r += dr
                check_c += dc
            
            # Bonus for staying away from borders
            border_penalty = 0
            if next_r < 2 or next_r >= GRID_HEIGHT - 2:
                border_penalty += 2
            if next_c < 2 or next_c >= GRID_WIDTH - 2:
                border_penalty += 2
            
            score -= border_penalty
            
            if score > best_score:
                best_score = score
                best_action = action
        
        return best_action
    
    def train_step(self):
        """Perform one training step."""
        if len(self.memory) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)
        
        # Current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Next Q values from target network
        next_q = self.target_net(next_states).max(1)[0]
        target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss
        loss = nn.MSELoss()(current_q, target_q.detach())
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
    
    def train(self, episodes=1000, save_path='model.pth'):
        """Train the agent."""
        print(f"Starting training for {episodes} episodes...")
        
        for episode in range(episodes):
            game = Game()
            total_reward = 0
            done = False
            
            while not done:
                # Agent 1 (learning agent)
                state1 = self.extract_features(game, 1)
                valid_actions1 = self.get_valid_actions(game.agent1)
                action1 = self.select_action(state1, valid_actions1, training=True)
                
                # Agent 2 (opponent)
                action2 = self.opponent_policy(game)
                
                # Execute moves
                dir1 = self.action_map[action1]
                dir2 = self.action_map[action2]
                result = game.step(dir1, dir2)
                
                # Calculate reward
                reward = 0
                if result == GameResult.AGENT1_WIN:
                    reward = 100
                    done = True
                elif result == GameResult.AGENT2_WIN:
                    reward = -100
                    done = True
                elif result == GameResult.DRAW:
                    reward = 0
                    done = True
                else:
                    # Reward for surviving
                    reward = 1
                    # Bonus for more trail
                    reward += (game.agent1.length - game.agent2.length) * 0.1
                
                next_state1 = self.extract_features(game, 1)
                self.memory.push(state1, action1, reward, next_state1, done)
                
                total_reward += reward
                
                # Train
                self.train_step()
                
                # Render
                if self.visualize and episode % 10 == 0:
                    self.render(game, episode, total_reward, self.epsilon)
            
            # Record stats
            self.episode_rewards.append(total_reward)
            won = 1 if result == GameResult.AGENT1_WIN else 0
            self.win_rate.append(won)
            
            # Update target network
            if episode % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Print progress
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:]) if len(self.episode_rewards) >= 100 else np.mean(self.episode_rewards)
                avg_win = np.mean(self.win_rate[-100:]) if len(self.win_rate) >= 100 else np.mean(self.win_rate)
                print(f"Episode {episode + 1}/{episodes} | Avg Reward: {avg_reward:.2f} | Win Rate: {avg_win:.2%} | Epsilon: {self.epsilon:.3f}")
            
            # Save model periodically
            if (episode + 1) % 100 == 0:
                torch.save(self.policy_net.state_dict(), save_path)
                print(f"Model saved to {save_path}")
        
        # Final save
        torch.save(self.policy_net.state_dict(), save_path)
        print(f"Training complete! Final model saved to {save_path}")
        
        if self.visualize:
            pygame.quit()

if __name__ == "__main__":
    # Train the agent
    trainer = TronTrainer(visualize=True)
    trainer.train(episodes=1000, save_path='model.pth')