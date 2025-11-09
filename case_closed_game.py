import random
from collections import deque
from enum import Enum
from typing import Optional

# top-level constants
EMPTY  = 0
AGENT1 = 1
AGENT2 = 2


class GameBoard:
    def __init__(self, height: int = 18, width: int = 20):
        self.height = height
        self.width = width
        self.grid = [[EMPTY for _ in range(width)] for _ in range(height)]

    def get_cell_state(self, position: tuple[int, int]) -> int:
        x, y = position
        return self.grid[y][x]

    def set_cell_state(self, position: tuple[int, int], state: int):
        x, y = position
        self.grid[y][x] = state

    def get_random_empty_cell(self) -> tuple[int, int] | None:
        empty_cells = []
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x] == EMPTY:
                    empty_cells.append((x, y))
        if not empty_cells:
            return None
        return random.choice(empty_cells)

    # <-- OUTDENTED: this was nested before; now it's a proper method
    def __str__(self) -> str:
        chars = {EMPTY: '.', AGENT1: 'A', AGENT2: 'B'}
        board_str = ""
        for y in range(self.height):
            for x in range(self.width):
                board_str += chars.get(self.grid[y][x], '?') + ' '
            board_str += '\n'
        return board_str


class Direction(Enum):
    UP    = (0, -1)
    DOWN  = (0,  1)
    RIGHT = (1,  0)
    LEFT  = (-1, 0)


class GameResult(Enum):
    AGENT1_WIN = 1
    AGENT2_WIN = 2
    DRAW       = 3


class Agent:
    """Represents an agent; manages its trail and movement."""
    def __init__(self, agent_id: str, start_pos: tuple[int, int],
                 start_dir: Direction, board: GameBoard, mark: int):
        self.agent_id = agent_id
        self.mark = mark                      # <— accept and store mark
        second = (start_pos[0] + start_dir.value[0], start_pos[1] + start_dir.value[1])
        self.trail = deque([start_pos, second])
        self.direction = start_dir
        self.board = board
        self.alive = True
        self.length = 2
        self.boosts_remaining = 3

        # paint initial cells with THIS agent's mark
        self.board.set_cell_state(start_pos, self.mark)
        self.board.set_cell_state(second,   self.mark)
    
    def is_head(self, position: tuple[int, int]) -> bool:
        return position == self.trail[-1]
    
    def move(self, direction: Direction, other_agent: Optional['Agent'] = None,
             use_boost: bool = False) -> bool:
        if not self.alive:
            return False

        if use_boost and self.boosts_remaining <= 0:
            # no boosts left; ignore
            use_boost = False
        
        num_moves = 2 if use_boost else 1
        if use_boost:
            self.boosts_remaining -= 1
            print(f'Agent {self.agent_id} used boost! ({self.boosts_remaining} remaining)')
        
        for _ in range(num_moves):
            # prevent immediate reversal
            cur_dx, cur_dy = self.direction.value
            req_dx, req_dy = direction.value
            if (req_dx, req_dy) == (-cur_dx, -cur_dy):
                print('invalid move (reverse); skipping this step')
                continue
            
            head = self.trail[-1]
            dx, dy = direction.value
            new_head = (head[0] + dx, head[1] + dy)
            
            # out of bounds => death
            if not (0 <= new_head[0] < self.board.width and 0 <= new_head[1] < self.board.height):
                self.alive = False
                return False
            
            cell_state = self.board.get_cell_state(new_head)
            self.direction = direction

            # collision with ANY trail (A or B)
            if cell_state != EMPTY:
                # own trail?
                if new_head in self.trail:
                    self.alive = False
                    return False
                # opponent trail?
                if other_agent and other_agent.alive and new_head in other_agent.trail:
                    if other_agent.is_head(new_head):  # head-on -> both die (draw)
                        self.alive = False
                        other_agent.alive = False
                        return False
                    else:  # hit opponent body
                        self.alive = False
                        return False

            # safe move – lay down THIS agent's mark
            self.trail.append(new_head)
            self.length += 1
            self.board.set_cell_state(new_head, self.mark)
        
        return True

    def get_trail_positions(self) -> list[tuple[int, int]]:
        return list(self.trail)
    

class Game:
    def __init__(self):
        self.board = GameBoard()
        self.agent1 = Agent(agent_id="1", start_pos=(1, 2),
                            start_dir=Direction.RIGHT, board=self.board, mark=AGENT1)
        self.agent2 = Agent(agent_id="2", start_pos=(17, 15),
                            start_dir=Direction.LEFT,  board=self.board, mark=AGENT2)
        self.turns = 0

    def reset(self):
        self.board = GameBoard()
        self.agent1 = Agent(agent_id="1", start_pos=(1, 2),
                            start_dir=Direction.RIGHT, board=self.board, mark=AGENT1)
        self.agent2 = Agent(agent_id="2", start_pos=(17, 15),
                            start_dir=Direction.LEFT,  board=self.board, mark=AGENT2)
        self.turns = 0

    def step(self, dir1: Direction, dir2: Direction,
             boost1: bool = False, boost2: bool = False):
        if self.turns >= 200:
            # length tiebreak
            if self.agent1.length > self.agent2.length:
                return GameResult.AGENT1_WIN
            elif self.agent2.length > self.agent1.length:
                return GameResult.AGENT2_WIN
            else:
                return GameResult.DRAW
        
        a1_alive = self.agent1.move(dir1, other_agent=self.agent2, use_boost=boost1)
        a2_alive = self.agent2.move(dir2, other_agent=self.agent1, use_boost=boost2)

        if not a1_alive and not a2_alive:
            return GameResult.DRAW
        elif not a1_alive:
            return GameResult.AGENT2_WIN
        elif not a2_alive:
            return GameResult.AGENT1_WIN

        self.turns += 1
        return None  # game continues
