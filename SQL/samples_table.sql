CREATE TABLE samples (
    id_hash TEXT PRIMARY KEY,
    fen TEXT NOT NULL,
    value REAL NOT NULL,
    move TEXT,
    winner TEXT,
    side_to_move CHAR(1),
    game_id TEXT,
    move_number INT,
    source TEXT,
    png TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
