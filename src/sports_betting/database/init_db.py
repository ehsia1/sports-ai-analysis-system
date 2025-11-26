"""Database initialization script."""

if __name__ == "__main__":
    from .session import init_db
    
    print("Initializing database...")
    init_db()
    print("Database initialized successfully!")