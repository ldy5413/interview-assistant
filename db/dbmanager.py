import os
import sqlite3

class DatabaseManager:
    def __init__(self, db_path: str | None = None):
        """Simple wrapper around SQLite used for storing transcriptions.

        Parameters
        ----------
        db_path: str | None
            Optional path to the database file. If not provided the value of the
            ``DB_PATH`` environment variable is used, falling back to
            ``transcriptions.db``.
        """
        self.db_path = db_path or os.getenv("DB_PATH", "transcriptions.db")
        # Create the table in the main thread
        self._init_db()
    
    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS transcriptions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
        finally:
            conn.close()
    
    def add_transcription(self, text):
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute('INSERT INTO transcriptions (text, timestamp) VALUES (?, datetime("now", "localtime"))', (text,))
            conn.commit()
            # Get the timestamp of the just-inserted row
            cursor.execute('SELECT timestamp FROM transcriptions WHERE id = last_insert_rowid()')
            timestamp = cursor.fetchone()[0]
            return timestamp
        finally:
            conn.close()
    
    def get_last_n_lines(self, n):
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute('SELECT timestamp, text FROM transcriptions ORDER BY id DESC LIMIT ?', (n,))
            return [(row[0], row[1]) for row in cursor.fetchall()]
        finally:
            conn.close()
    
    def close(self):
        # No need to close a persistent connection anymore
        pass
