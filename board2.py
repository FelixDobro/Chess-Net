import pandas as pd
import chess.pgn
import io
import os
import numpy as np
import json
import chess

# Konfiguration
CHUNK_SIZE = 50000
OUTPUT_DIR = "new_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Lade Move-Encoding
with open("uci_moves.json", "r") as f:
    moves = json.load(f)
move_to_idx = {move: i for i, move in enumerate(moves)}

# Board zu Tensor konvertieren (deine Funktion)

def board_to_tensor(board: chess.Board, history_length: int = 8):
    piece_map = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }

    planes_per_board = 12
    extra_planes = 6  # turn + castling (4) + move clock
    total_planes = history_length * planes_per_board + extra_planes
    tensor = np.zeros((total_planes, 8, 8), dtype=np.float32)

    # Copy board and extract history by popping moves
    board_copy = board.copy(stack=False)
    boards = []
    for _ in range(history_length):
        boards.append(board_copy.copy(stack=False))
        if board_copy.move_stack:
            board_copy.pop()
        else:
            break
    boards = boards[::-1]  # oldest → newest

    # Fill planes with piece positions
    for h_idx, past_board in enumerate(boards):
        for square in chess.SQUARES:
            piece = past_board.piece_at(square)
            if piece:
                row = 7 - chess.square_rank(square)
                col = chess.square_file(square)
                offset = h_idx * planes_per_board
                idx = piece_map[piece.piece_type]
                if piece.color == chess.WHITE:
                    tensor[offset + idx, row, col] = 1
                else:
                    tensor[offset + idx + 6, row, col] = 1

    # Extra planes
    tensor[-6, :, :] = 1.0 if board.turn == chess.WHITE else 0.0
    tensor[-5, :, :] = int(board.has_kingside_castling_rights(chess.WHITE))
    tensor[-4, :, :] = int(board.has_queenside_castling_rights(chess.WHITE))
    tensor[-3, :, :] = int(board.has_kingside_castling_rights(chess.BLACK))
    tensor[-2, :, :] = int(board.has_queenside_castling_rights(chess.BLACK))
    tensor[-1, :, :] = board.fullmove_number / 100.0  # normalized

    return tensor

# Listen zum Puffern
boards, moves_idx, values = [], [], []
file_idx = 0

# CSV einlesen und verarbeiten
for chunk in pd.read_csv("data/chess_games.csv", chunksize=100000):
    filtered = chunk[(chunk["WhiteElo"] + chunk["BlackElo"]) >= 4700]

    for moves_string in filtered["AN"].values:
        game = chess.pgn.read_game(io.StringIO(moves_string))
        if not game:
            continue

        board = game.board()
        for move in game.mainline_moves():
            tensor = board_to_tensor(board)
            move_idx = move_to_idx[move.uci()]
            if move_idx is not None:
                boards.append(tensor)
                moves_idx.append(move_idx)

            board.push(move)

            # Falls genug Daten für ein File:
            if len(boards) >= CHUNK_SIZE:
                filename = os.path.join(OUTPUT_DIR, f"chunk_{file_idx:05d}.npz")
                np.savez_compressed(filename, boards=np.array(boards), moves=np.array(moves_idx), values=np.array(values))
                print(f"💾 Gespeichert: {filename} mit {len(boards)} Beispielen")
                boards, moves_idx = [], []
                file_idx += 1

# Reste abspeichern
if boards:
    filename = os.path.join(OUTPUT_DIR, f"chunk_{file_idx:05d}.npz")
    np.savez_compressed(filename, boards=np.array(boards), moves=np.array(moves_idx), values=np.array(values))
    print(f"💾 Gespeichert: {filename} mit {len(boards)} Beispielen")
