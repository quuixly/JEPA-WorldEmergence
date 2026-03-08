from game.othello import GameBoard, Piece
import numpy as np


def test_default_pieces():
    game_board = GameBoard()
    row, col = game_board.position_to_index("E4")
    assert game_board.get_board()[row, col] == Piece.BLACK

    row, col = game_board.position_to_index("D5")
    assert game_board.get_board()[row, col] == Piece.BLACK

    row, col = game_board.position_to_index("E5")
    assert game_board.get_board()[row, col] == Piece.WHITE

    row, col = game_board.position_to_index("D4")
    assert game_board.get_board()[row, col] == Piece.WHITE

def test_restore_game_history():
    history = [
        (Piece.BLACK, "D3"),
        (Piece.WHITE, "C3"),
        (Piece.BLACK, "C4")
    ]

    game_board = GameBoard(history)
    board = game_board.get_board()

    row, col = game_board.position_to_index("D3")
    assert board[row, col] == Piece.BLACK

    row, col = game_board.position_to_index("D4")
    assert board[row, col] == Piece.BLACK

    row, col = game_board.position_to_index("C4")
    assert board[row, col] == Piece.BLACK

    row, col = game_board.position_to_index("C3")
    assert board[row, col] == Piece.WHITE

def test_restore_game_history_with_pass():
    moves = [
        (Piece.BLACK, "F5"), (Piece.WHITE, "D6"),
        (Piece.BLACK, "C5"), (Piece.WHITE, "B6"),
        (Piece.BLACK, "C3"), (Piece.WHITE, "D3"),
        (Piece.BLACK, "C7"), (Piece.WHITE, "G6"),
        (Piece.BLACK, "A5"), (Piece.WHITE, "F6"),
        (Piece.BLACK, "E3"), (Piece.WHITE, "B2"),
        (Piece.BLACK, "B3"), (Piece.WHITE, "B4"),
        (Piece.BLACK, "F4"), (Piece.WHITE, "F3"),
        (Piece.BLACK, "F2"), (Piece.WHITE, "F1"),
        (Piece.BLACK, "D2"), (Piece.WHITE, "D1"),
        (Piece.BLACK, "E1"), (Piece.WHITE, "B5"),
        (Piece.BLACK, "A1"), (Piece.WHITE, "E6"),
        (Piece.BLACK, "G3"), (Piece.WHITE, "G4"),
        (Piece.BLACK, "E7"), (Piece.WHITE, "H3"),
        (Piece.BLACK, "C4"), (Piece.WHITE, "F7"),
        (Piece.BLACK, "G7"), (Piece.WHITE, "A7"),
        (Piece.BLACK, "C1"), (Piece.WHITE, "E8"),
        (Piece.BLACK, "E2"), (Piece.WHITE, "G1"),
        (Piece.BLACK, "H7"), (Piece.WHITE, "C6"),
        (Piece.BLACK, "B7"), (Piece.WHITE, "H6"),
        (Piece.BLACK, "H1"), (Piece.WHITE, "A4"),
        (Piece.BLACK, "D7"), (Piece.WHITE, "A3"),
        (Piece.BLACK, "H5"), (Piece.WHITE, "B1"),
        (Piece.BLACK, "A2"), (Piece.WHITE, "C2"),
        (Piece.BLACK, "H2"), (Piece.WHITE, "B8"),
        (Piece.BLACK, "A6"), (Piece.WHITE, "D8"),
        (Piece.BLACK, "A8"), (Piece.WHITE, "C8"),
        (Piece.BLACK, "F8"), (Piece.WHITE, "G5"),
        (Piece.BLACK, "H4"), (Piece.WHITE, "G2"), # Pass here, next move white
        (Piece.WHITE, "G8"), (Piece.BLACK, "H8")
    ]

    target_board = np.array([
        [Piece.BLACK, Piece.WHITE, Piece.BLACK, Piece.BLACK, Piece.BLACK, Piece.BLACK, Piece.BLACK, Piece.BLACK],
        [Piece.BLACK, Piece.WHITE, Piece.WHITE, Piece.WHITE, Piece.WHITE, Piece.WHITE, Piece.WHITE, Piece.BLACK],
        [Piece.BLACK, Piece.BLACK, Piece.WHITE, Piece.WHITE, Piece.WHITE, Piece.WHITE, Piece.WHITE, Piece.BLACK],
        [Piece.BLACK, Piece.BLACK, Piece.BLACK, Piece.WHITE, Piece.WHITE, Piece.WHITE, Piece.WHITE, Piece.BLACK],
        [Piece.BLACK, Piece.BLACK, Piece.BLACK, Piece.WHITE, Piece.WHITE, Piece.WHITE, Piece.WHITE, Piece.BLACK],
        [Piece.BLACK, Piece.BLACK, Piece.WHITE, Piece.BLACK, Piece.WHITE, Piece.BLACK, Piece.WHITE, Piece.BLACK],
        [Piece.BLACK, Piece.BLACK, Piece.WHITE, Piece.WHITE, Piece.BLACK, Piece.WHITE, Piece.BLACK, Piece.BLACK],
        [Piece.BLACK, Piece.BLACK, Piece.BLACK, Piece.BLACK, Piece.BLACK, Piece.BLACK, Piece.BLACK, Piece.BLACK]
    ], dtype=object)

    game_board = GameBoard(moves)
    assert np.array_equal(game_board.get_board(), target_board)

def test_legal_moves():
    history = [
        (Piece.BLACK, "D3"),
        (Piece.WHITE, "C3"),
        (Piece.BLACK, "C4"),
        (Piece.WHITE, "E3"),
    ]

    game_board = GameBoard(history)

    legal_moves = game_board.get_legal_moves(Piece.BLACK)
    true_legal_moves = ("B2", "C2", "D2", "E2", "F2", "F3", "F4", "F5", "F6")

    for move in true_legal_moves:
        assert move in legal_moves

    legal_moves = game_board.get_legal_moves(Piece.WHITE)
    true_legal_moves = ("B4", "B5", "C5", "C6", "D6")

    for move in true_legal_moves:
        assert move in legal_moves

def test_legal_moves_at_the_end():
    game_end_history = [
        (Piece.BLACK, "E6"),
        (Piece.WHITE, "F4"),
        (Piece.BLACK, "F6"),
        (Piece.WHITE, "G5"),
        (Piece.BLACK, "D6"),
        (Piece.WHITE, "E7"),
        (Piece.BLACK, "F5"),
        (Piece.WHITE, "C5")
    ]
    game_end_board = GameBoard(game_end_history)

    legal_moves = game_end_board.get_legal_moves(Piece.BLACK)
    true_legal_moves = ()

    for move in legal_moves:
        assert move not in true_legal_moves

    legal_moves = game_end_board.get_legal_moves(Piece.WHITE)
    true_legal_moves = ()

    for move in legal_moves:
        assert move not in true_legal_moves

def test_simulate_game():
    for _ in range(5):
        game = GameBoard()
        game.simulate_game()
        assert game.get_legal_moves(Piece.BLACK) == []
        assert game.get_legal_moves(Piece.WHITE) == []