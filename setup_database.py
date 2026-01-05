"""
Database setup script for SQLite
Run this first to set up the SQLite database
"""
from database import Database
from config import Config
import os


def setup_database():
    """Set up SQLite database - no installation required"""
    print("=" * 60)
    print("SQLite Database Setup")
    print("=" * 60)
    
    try:
        db = Database()
        if not db.connect():
            print("\n[ERROR] Setup failed")
            return False
        
        db.create_extension()  # Dummy call for compatibility
        db.create_table()
        db.close()
        
        return True
    except Exception as e:
        print(f"[ERROR] Error setting up database: {e}")
        return False


def main():
    """Main setup function"""
    print("\n[SETUP] Setting up SQLite database...")
    print("[INFO] No installation required - SQLite comes with Python!")
    print()
    
    if not setup_database():
        print("\n[ERROR] Setup failed")
        return
    
    print("\n" + "=" * 60)
    print("[SUCCESS] Setup completed successfully!")
    print("=" * 60)
    print("\nDatabase file created: rag_database.db")
    print("\nNext steps:")
    print("1. Install Python dependencies: pip install -r requirements.txt")
    print("2. Install Ollama: https://ollama.ai")
    print("3. Pull a model: ollama pull mistral")
    print("4. Run: python main.py")


if __name__ == "__main__":
    main()
