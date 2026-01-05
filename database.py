"""
SQLite database module - NO INSTALLATION REQUIRED
Uses SQLite which comes with Python by default
"""
import sqlite3
from typing import List, Tuple, Optional
import json
import math
import os
from config import Config


class Database:
    """SQLite database handler with manual vector similarity"""
    
    def __init__(self):
        self.config = Config
        self.conn: Optional[sqlite3.Connection] = None
        self.cur: Optional[sqlite3.Cursor] = None
        self.db_path = Config.DB_FILE
    
    def connect(self):
        """Establish database connection"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cur = self.conn.cursor()
            print(f"[SUCCESS] Connected to SQLite database: {self.db_path}")
            return True
        except sqlite3.Error as e:
            print(f"[ERROR] Database connection error: {e}")
            return False
    
    def create_extension(self):
        """Dummy method for compatibility - SQLite doesn't need extensions"""
        print("[INFO] Using SQLite (no extensions needed)")
        return True
    
    def create_table(self):
        """Create documents table"""
        if not self.conn or not self.cur:
            raise RuntimeError("Database not connected. Call connect() first.")
        
        try:
            create_table_query = """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                embedding TEXT NOT NULL,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
            self.cur.execute(create_table_query)
            
            # Create index for faster retrieval
            index_query = """
            CREATE INDEX IF NOT EXISTS idx_documents_id ON documents(id);
            """
            self.cur.execute(index_query)
            
            self.conn.commit()
            print("[SUCCESS] Documents table created (SQLite - no installation needed!)")
            print("[INFO] Using manual similarity calculation")
            return True
        except sqlite3.Error as e:
            print(f"[ERROR] Error creating table: {e}")
            return False
    
    def insert_document(self, content: str, embedding: List[float], metadata: Optional[dict] = None):
        """Insert a single document with its embedding"""
        if not self.conn or not self.cur:
            raise RuntimeError("Database not connected. Call connect() first.")
        
        try:
            embedding_json = json.dumps(embedding)
            metadata_json = json.dumps(metadata) if metadata else None
            
            insert_query = """
            INSERT INTO documents (content, embedding, metadata)
            VALUES (?, ?, ?)
            """
            self.cur.execute(insert_query, (content, embedding_json, metadata_json))
            self.conn.commit()
            return self.cur.lastrowid
        except sqlite3.Error as e:
            print(f"[ERROR] Error inserting document: {e}")
            return None
    
    def insert_documents_batch(self, documents: List[Tuple[str, List[float], dict]]):
        """Insert multiple documents in batch"""
        if not self.conn or not self.cur:
            raise RuntimeError("Database not connected. Call connect() first.")
        
        try:
            prepared_docs = [
                (content, json.dumps(embedding), json.dumps(metadata) if metadata else None)
                for content, embedding, metadata in documents
            ]
            
            insert_query = """
            INSERT INTO documents (content, embedding, metadata)
            VALUES (?, ?, ?)
            """
            self.cur.executemany(insert_query, prepared_docs)
            self.conn.commit()
            print(f"[SUCCESS] Inserted {len(documents)} documents")
            return list(range(1, len(documents) + 1))  # Return mock IDs
        except sqlite3.Error as e:
            print(f"[ERROR] Error inserting documents: {e}")
            return None
    
    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def similarity_search(self, query_embedding: List[float], k: int = 5) -> List[Tuple[str, float]]:
        """
        Perform vector similarity search using manual calculation
        Returns top k most similar documents
        """
        if not self.conn or not self.cur:
            raise RuntimeError("Database not connected. Call connect() first.")
        
        try:
            # Limit query to reasonable number to avoid memory issues
            # For production, consider implementing approximate nearest neighbor search
            self.cur.execute("SELECT id, content, embedding FROM documents LIMIT 10000")
            all_docs = self.cur.fetchall()
            
            if not all_docs:
                return []
            
            similarities = []
            for doc_id, content, embedding_str in all_docs:
                embedding = json.loads(embedding_str)
                similarity = self.cosine_similarity(query_embedding, embedding)
                distance = 1 - similarity
                similarities.append((content, distance))
            
            similarities.sort(key=lambda x: x[1])
            return similarities[:k]
            
        except sqlite3.Error as e:
            print(f"[ERROR] Error performing similarity search: {e}")
            return []
        except json.JSONDecodeError as e:
            print(f"[ERROR] Error decoding embedding: {e}")
            return []
    
    def get_document_count(self) -> int:
        """Get total number of documents"""
        if not self.conn or not self.cur:
            raise RuntimeError("Database not connected. Call connect() first.")
        
        try:
            self.cur.execute("SELECT COUNT(*) FROM documents")
            count = self.cur.fetchone()[0]
            return count
        except sqlite3.Error as e:
            print(f"[ERROR] Error getting document count: {e}")
            return 0
    
    def clear_all_documents(self):
        """Clear all documents from the table"""
        if not self.conn or not self.cur:
            raise RuntimeError("Database not connected. Call connect() first.")
        
        try:
            self.cur.execute("DELETE FROM documents")
            self.cur.execute("DELETE FROM sqlite_sequence WHERE name='documents'")
            self.conn.commit()
            print("[SUCCESS] All documents cleared")
            return True
        except sqlite3.Error as e:
            print(f"[ERROR] Error clearing documents: {e}")
            return False
    
    def close(self):
        """Close database connection"""
        if self.cur:
            self.cur.close()
        if self.conn:
            self.conn.close()
        print("[INFO] Database connection closed")
    
    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
