import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import hashlib
import psycopg2
import pandas as pd
from psycopg2.extras import execute_values
from config import config

def main():
    def parse_eval(raw):
        try:
            raw = str(raw).strip()

            if raw.startswith("#"):
                sign = -1 if "-" in raw else 1
                return 10001 * sign
            return float(raw)
        except:
            return None


    def insert_chunk(cursor, rows):
        query = """
            INSERT INTO samples (
                id_hash, fen, value, move, winner, side_to_move,
                game_id, move_number, source, png
            )
            VALUES %s
            ON CONFLICT (id_hash) DO NOTHING
        """
        execute_values(cursor, query, rows)


    def make_id_hash(fen, value, move=None):
        base = f"{fen}_{value}_{move or ''}"
        return hashlib.sha256(base.encode()).hexdigest()


    def prepare_rows(df):
        rows = []
        for _, row in df.iterrows():
            fen = row["FEN"]
            value = parse_eval(row["Evaluation"])
            move = row.get("Move", None)
            id_hash = make_id_hash(fen, value, move)

            rows.append((
                id_hash,
                fen,
                value,
                row.get("move", None),
                row.get("winner", None),
                row.get("side_to_move", None),
                row.get("game_id", None),
                row.get("move_number", None),
                row.get("source", None),
                row.get("png", None),
            ))
        return rows


    conn = psycopg2.connect(
        dbname="chessdb", user="chess", password="secret", host="localhost", port=5440
    )


    chunks = pd.read_csv(config["data_path"], chunksize=10000)
    for i, chunk in enumerate(chunks):
        rows = prepare_rows(chunk)
        with conn:
            with conn.cursor() as cur:
                insert_chunk(cur, rows)
                print(f"{(i+1)* 10000} chunks saved")

if __name__ == "__main__":
    main()