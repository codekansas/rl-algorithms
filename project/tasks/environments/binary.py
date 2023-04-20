import math
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
    last_action: Action | None


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
        start_tiles: int = 5,
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
        return State(self.board.copy(), 0, False, None)

    def render(self, state: State) -> np.ndarray:
        # Draws the board to an RGB image where each tile has shape (N, N, 3)
        # and contains the value of the tile in the center of the tile,
        # converted from the power to the actual number (i.e., 11 -> 2048).

        tile_size = 32
        half_t = tile_size // 2
        board = state.board
        pixels_w, pixels_h = board.shape[0] * tile_size, board.shape[1] * tile_size

        # Creates the colors.
        image_arr = np.zeros((*board.shape, tile_size, tile_size, 3), dtype=np.uint8)
        for i in range(board.shape[0]):
            for j in range(board.shape[1]):
                color = COLORS[board[i, j]]
                image_arr[i, j, :, :] = color
        image_arr = image_arr.transpose(0, 2, 1, 3, 4)
        image_arr = image_arr.reshape(tile_size * board.shape[0], tile_size * board.shape[1], 3)
        image = PIL.Image.fromarray(image_arr)

        # Draws the text.
        font = PIL.ImageFont.load_default()
        draw = PIL.ImageDraw.Draw(image)
        for i in range(board.shape[0]):
            for j in range(board.shape[1]):
                if board[i, j] == 0:
                    continue
                value = 2 ** board[i, j]
                if value > 4:
                    text_color = (249, 246, 242)
                else:
                    text_color = (119, 110, 101)
                bbox = font.getbbox(str(value))
                x = half_t + j * tile_size - bbox[2] // 2
                y = half_t + i * tile_size - bbox[3] // 2
                draw.text((x, y), str(value), fill=text_color, font=font)

        # Draws an arrow and text caption indicating the last action.
        if state.last_action is not None:
            rot = {"up": 3, "right": 0, "down": 1, "left": 2}[state.last_action] * math.pi / 2
            arc = math.pi / 6
            diam_o, diam_i = 16, 12
            points = [
                (math.cos(rot + arc) * diam_i, math.sin(rot + arc) * diam_i),
                (math.cos(rot - arc) * diam_i, math.sin(rot - arc) * diam_i),
                (math.cos(rot) * diam_o, math.sin(rot) * diam_o),
            ]
            points = [(pixels_w / 2 + p[0], pixels_h / 2 + p[1]) for p in points]
            draw.polygon(points, fill=(255, 255, 255), outline=(0, 0, 0), width=1)

            action = str(state.last_action)[:1].upper()
            bbox = font.getbbox(action)
            x = pixels_w / 2 - bbox[2] // 2
            y = pixels_h / 2 - bbox[3] // 2
            text_color = (119, 110, 101)
            draw.text((x, y), action, fill=text_color, font=font)

        return np.array(image)

    def sample_action(self) -> Action:
        return np.random.choice(["up", "down", "left", "right"])

    def step(self, action: Action) -> State:
        # Moves the tiles in the chosen direction.
        rot = {"up": 1, "right": 2, "down": 3, "left": 0}[action]
        board = self.board.copy()
        board = np.rot90(board, rot)

        # Gets a new board by sliding the tiles.
        new_board = np.zeros_like(board)
        for i in range(board.shape[0]):
            row = board[i, :]
            row = row[row != 0]
            items: list[int] = []
            unpaired = True
            for r in row:
                if items and items[-1] == r and unpaired:
                    items[-1] += 1
                    unpaired = False
                else:
                    items.append(r)
                    unpaired = True
            new_board[i, : len(items)] = items

        # Rotates the board back to its original orientation.
        board = np.rot90(new_board, -rot)

        # Adds new tiles to the board.
        self.board = board
        # terminated = self._add_tiles(self.tiles_per_turn)
        terminated = False

        # Updates the score.
        score = int(np.sum(2**self.board))

        return State(self.board.copy(), score, terminated, action)

    def terminated(self, state: State) -> bool:
        return state.terminated or state.board.max() >= self.win_score

    @property
    def fps(self) -> int:
        return 5
