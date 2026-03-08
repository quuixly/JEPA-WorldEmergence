from dataset.generator import DatasetGenerator
from game.othello import GameBoard, Piece
import pytest
import torch


@pytest.fixture
def dataset():
    return DatasetGenerator()

def test_convert_game_to_tensor(dataset):
    history = [
        (Piece.BLACK, "F5"),
        (Piece.WHITE, "F4"),
        (Piece.BLACK, "C3")
    ]

    game_board = GameBoard(history)
    target = torch.full((60, 1), dataset.padding_token)
    target[0] = 34
    target[1] = 28
    target[2] = 19

    assert torch.all(dataset.convert_game_to_tensor(game_board) == target)
