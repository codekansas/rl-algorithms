from dataclasses import dataclass
from typing import Literal

import ml.api as ml
import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont

Action = Literal["up", "down", "left", "right"]


@dataclass
class State:
    board: np.ndarray
    score: int
    terminated: bool


COLORS: list[tuple[int, int, int]] = [
    (205, 193, 180),  # 0
    (238, 228, 218),  # 2
    (237, 224, 200),  # 4
    (242, 177, 121),  # 8
    (245, 149, 99),  # 16
    (246, 124, 95),  # 32
    (246, 94, 59),  # 64
    (237, 207, 114),  # 128
    (237, 204, 97),  # 256
    (237, 200, 80),  # 512
    (237, 197, 63),  # 1024
    (237, 194, 46),  # 2048
    (60, 58, 50),
    (60, 58, 50),
    (60, 58, 50),
    (60, 58, 50),
]


class Environment(ml.Environment[State, Action]):
    def __init__(
        self,
        shape: tuple[int, int] = (4, 4),
        win_score: int = 11,  # 2048
        start_tiles: int = 2,
        tiles_per_turn: int = 1,
    ) -> None:
        """Defines the 2048 game environment.

        The rules of the game are as follows:

        - The board is a 4x4 grid of tiles.
        - Each tile has a value of 2^n, where n is an integer.
        - The goal is to get a tile with a value of 2048.
        - The game starts with two tiles, each with a value of 2.
        - Each turn, a new tile is added to the board.
        - The new tile can be a 2 or a 4, with a 90% chance of being a 2.
        - The new tile is added to a random empty space on the board.
        - The player can move the tiles up, down, left, or right.
        - When the player moves the tiles, all tiles slide as far as possible
          in the chosen direction.
        - When two tiles with the same value collide while moving, they merge
          into a tile with the combined value of the two tiles that collided.
        - The resulting tile cannot merge with another tile again in the same
          move.
        - If there are no empty spaces on the board and no legal moves, the
          game ends.
        - The score is the sum of the values of all tiles on the board.

        Args:
            shape: The shape of the board.
            win_score: The score required to win the game.
            start_tiles: The number of tiles to start the game with.
            tiles_per_turn: The number of tiles to add per turn.
        """

        super().__init__()

        self.shape = shape
        self.win_score = win_score
        self.start_tiles = start_tiles
        self.tiles_per_turn = tiles_per_turn

        self.board = np.zeros(shape, dtype=np.int8)
        self.score = 0

    def _add_tiles(self, n: int = 1) -> bool:
        for _ in range(n):
            tile = np.random.choice([2, 4], p=[0.9, 0.1])
            if not (self.board == 0).any():
                return True  # Terminated
            ijs = np.argwhere(self.board == 0)
            i, j = ijs[np.random.choice(len(ijs))]
            self.board[i, j] = tile
        return False

    def reset(self, seed: int | None = None) -> State:
        self.board.fill(0)
        self._add_tiles(self.start_tiles)
        self.score = 0
        return State(self.board.copy(), self.score, False)

    def render(self, state: State) -> np.ndarray:
        # Draws the board to an RGB image where each tile has shape (64, 64, 3)
        # and contains the value of the tile in the center of the tile,
        # converted from the power to the actual number (i.e., 11 -> 2048).

        board = state.board

        # Creates the colors.
        image_arr = np.zeros((*board.shape, 64, 64, 3), dtype=np.uint8)
        for i in range(board.shape[0]):
            for j in range(board.shape[1]):
                color = COLORS[board[i, j]]
                image_arr[i, j, :, :] = color
        image_arr = image_arr.transpose(0, 2, 1, 3, 4).reshape(64 * board.shape[0], 64 * board.shape[1], 3)
        image = PIL.Image.fromarray(image_arr.reshape(256, 256, 3))

        # Draws the text.
        font = PIL.ImageFont.load_default()
        draw = PIL.ImageDraw.Draw(image)
        for i in range(board.shape[0]):
            for j in range(board.shape[1]):
                value = 2 ** board[i, j]
                if value > 4:
                    text_color = (249, 246, 242)
                else:
                    text_color = (119, 110, 101)
                tsize = font.getsize(str(value))
                x = 32 + j * 64 - tsize[0] // 2
                y = 32 + i * 64 - tsize[1] // 2
                draw.text((x, y), str(value), fill=text_color, font=font)

        return np.array(image)

    def sample_action(self) -> Action:
        return np.random.choice(["up", "down", "left", "right"])

    def step(self, action: Action) -> State:
        # Moves the tiles in the chosen direction.
        board = self.board.copy()
        if action == "up":
            board = np.rot90(board, k=3)
        elif action == "down":
            board = np.rot90(board)
        elif action == "left":
            pass
        elif action == "right":
            board = np.fliplr(board)

        # Slides the tiles as far as possible in the chosen direction.
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if board[i, j] != 0:
                    for k in range(1, self.shape[1] - j):
                        if board[i, j + k] != 0:
                            if board[i, j + k] == board[i, j]:
                                board[i, j] += 1
                                board[i, j + k] = 0
                            break
                        else:
                            board[i, j + k] = board[i, j]
                            board[i, j] = 0

        # Adds new tiles to the board.
        self.board = board
        terminated = self._add_tiles(self.tiles_per_turn)

        # Updates the score.
        self.score = int(np.sum(2**self.board))

        return State(self.board.copy(), self.score, terminated)

    def terminated(self, state: State) -> bool:
        return state.terminated or state.board.max() >= self.win_score
