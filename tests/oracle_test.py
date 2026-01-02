import os
import sys
import oracledb
from dotenv import load_dotenv
from pathlib import Path

# Add the backend directory to sys.path to allow imports if needed
backend_dir = Path(__file__).parent.parent / "src" 
sys.path.append(str(backend_dir))

# Load environment variables from the backend .env file
env_path = backend_dir / ".env"
load_dotenv(dotenv_path=env_path)

def test_oracle_connection():
    # Use environment variables or fallback to defaults from the .env we just loaded
    user = os.getenv("DATABASE_USER") or os.getenv("ADMIN_USERNAME") or "apps"
    password = os.getenv("DATABASE_PASSWORD") or os.getenv("ADMIN_PASSWORD") or "apps"
    dsn = os.getenv("DATABASE_DSN", "183.82.4.173:1529/VIS")
    lib_dir = os.getenv("ORACLE_CLIENT_LIB_UBUNTU", "/llamaSFT/askGuru-SQL/instantclient_23_9")

    print(f"--- Oracle Connection Test ---")
    print(f"DSN: {dsn}")
    print(f"User: {user}")
    
    # Initialize Oracle Client if lib_dir is provided and exists
    if lib_dir and os.path.exists(lib_dir):
        try:
            print(f"Initializing Oracle client from: {lib_dir}")
            oracledb.init_oracle_client(lib_dir=lib_dir)
            print("Oracle client initialized successfully.")
        except Exception as e:
            print(f"Warning: Could not initialize Oracle client: {e}")
            print("Attempting to connect in Thin mode...")
    else:
        print("Oracle client library path not found or not provided. Using Thin mode.")

    connection = None
    try:
        # Establish connection
        print("Connecting to database...")
        connection = oracledb.connect(
            user=user,
            password=password,
            dsn=dsn
        )
        print("Successfully connected to Oracle Database!")

        # Create a cursor and execute a simple query
        cursor = connection.cursor()
        
        # Test query
        query = "SELECT 'Connection Successful' FROM DUAL"
        print(f"Executing test query: {query}")
        cursor.execute(query)
        
        row = cursor.fetchone()
        if row:
            print(f"Result: {row[0]}")
        
        # Another test: Get database version
        print("Database version:", connection.version)
        
        cursor.close()
        return True

    except Exception as e:
        print(f"Error connecting to Oracle: {e}")
        return False
    finally:
        if connection:
            connection.close()
            print("Connection closed.")

if __name__ == "__main__":
    success = test_oracle_connection()
    if success:
        sys.exit(0)
    else:
        sys.exit(1)
