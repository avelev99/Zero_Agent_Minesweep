import numpy as np
import gymnasium as gym
from gymnasium import spaces


def _neighbors(pos, size):
    r, c = pos
    rows, cols = size
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                yield nr, nc


class MinesweeperEnv(gym.Env):
    """Minesweeper environment supporting reveal, flag, and chord actions."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, size=(8, 8), n_mines=10, seed=None, first_click_safe=True):
        super().__init__()
        self.size = size
        self.n_mines = n_mines
        self.first_click_safe = first_click_safe
        self.action_space = spaces.Discrete(size[0] * size[1] * 3)
        self.observation_space = spaces.Box(
            low=-2, high=9, shape=size, dtype=np.int8
        )
        self.rng = np.random.default_rng(seed)
        self.board = None
        self.visible = None
        self.first_move = True

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.board = None
        self.visible = -np.ones(self.size, dtype=np.int8)
        self.first_move = True
        return self.visible.copy(), {}

    # ------------------------------------------------------------------
    def _generate_board(self, safe_cell=None):
        rows, cols = self.size
        cells = rows * cols
        mines = np.zeros(cells, dtype=bool)
        safe_set = set()
        if safe_cell is not None:
            safe_set.add(safe_cell)
            safe_set.update(_neighbors(safe_cell, self.size))
        choices = [i for i in range(cells) if divmod(i, cols) not in safe_set]
        if self.n_mines > 0:
            mine_idx = self.rng.choice(choices, self.n_mines, replace=False)
            mines[mine_idx] = True
        else:
            mine_idx = np.array([], dtype=int)
        board = np.zeros(cells, dtype=np.int8)
        for idx in mine_idx:
            r, c = divmod(idx, cols)
            board[idx] = 9
            for nr, nc in _neighbors((r, c), self.size):
                if board[nr * cols + nc] != 9:
                    board[nr * cols + nc] += 1
        self.board = board.reshape(self.size)

    # ------------------------------------------------------------------
    def _reveal(self, r, c):
        if self.visible[r, c] >= 0:
            return
        stack = [(r, c)]
        while stack:
            cr, cc = stack.pop()
            if self.visible[cr, cc] >= 0 or self.visible[cr, cc] == -2:
                continue
            self.visible[cr, cc] = self.board[cr, cc]
            if self.board[cr, cc] == 0:
                for nr, nc in _neighbors((cr, cc), self.size):
                    if self.visible[nr, nc] == -1:
                        stack.append((nr, nc))

    # ------------------------------------------------------------------
    def step(self, action):
        rows, cols = self.size
        cells = rows * cols
        idx = action % cells
        act = action // cells  # 0=reveal, 1=flag, 2=chord
        r, c = divmod(idx, cols)
        reward = -0.01
        terminated = False
        if act == 1:
            if self.visible[r, c] == -1:
                self.visible[r, c] = -2
            elif self.visible[r, c] == -2:
                self.visible[r, c] = -1
            return self.visible.copy(), reward, terminated, False, {}
        if act == 2:
            if self.visible[r, c] >= 0 and self.board is not None:
                flagged = sum(
                    1 for nr, nc in _neighbors((r, c), self.size) if self.visible[nr, nc] == -2
                )
                if flagged == self.board[r, c]:
                    for nr, nc in _neighbors((r, c), self.size):
                        if self.visible[nr, nc] == -1:
                            if self.board[nr, nc] == 9:
                                self.visible[nr, nc] = 9
                                reward = -1.0
                                terminated = True
                                break
                            self._reveal(nr, nc)
                    if not terminated and np.all((self.visible >= 0) | (self.board == 9)):
                        reward = 1.0
                        terminated = True
            return self.visible.copy(), reward, terminated, False, {}
        if self.first_move:
            if self.first_click_safe:
                self._generate_board((r, c))
            else:
                self._generate_board()
            self.first_move = False
        if self.board[r, c] == 9:
            self.visible[r, c] = 9
            reward = -1.0
            terminated = True
        else:
            self._reveal(r, c)
            if np.all((self.visible >= 0) | (self.board == 9)):
                reward = 1.0
                terminated = True
        return self.visible.copy(), reward, terminated, False, {}

    # ------------------------------------------------------------------
    def action_mask(self):
        """Return a boolean mask of valid actions for the current state."""

        rows, cols = self.size
        cells = rows * cols
        mask = np.zeros(self.action_space.n, dtype=bool)

        hidden = self.visible == -1
        mask[:cells] = hidden.reshape(-1)

        flaggable = (self.visible == -1) | (self.visible == -2)
        mask[cells : 2 * cells] = flaggable.reshape(-1)

        chordable = self.visible >= 0
        mask[2 * cells : 3 * cells] = chordable.reshape(-1)

        return mask

    # ------------------------------------------------------------------
    def render(self):
        print(self.visible)


def make_vec_env(num_envs, **kwargs):
    """Create a vectorized Minesweeper environment."""

    def _make():
        return MinesweeperEnv(**kwargs)

    return gym.vector.SyncVectorEnv([_make for _ in range(num_envs)])
