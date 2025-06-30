import os
import uuid
import sqlite3
from typing import List, Tuple, Dict, Any
import chromadb
from chromadb.utils import embedding_functions

from logger.logging import logger

# Sqlite Datbase and ChromaDB Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "memory.db")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_data")

os.makedirs(CHROMA_DIR, exist_ok=True)

def get_db_connection():
    """
    Establishes and returns a connection to the SQLite database.
    
    Returns:
        A sqlite3.Connection object connected to memory.db.
    """
    return sqlite3.connect(DB_PATH)

def init_sqlite():
    """Initializes the SQLite database and creates the "prompts" table if it doesn't exist."""
    try:
        conn = get_db_connection()
        conn.row_factory = sqlite3.Row  # Allows accessing columns by name
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS prompts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                user_prompt TEXT NOT NULL,
                enhanced_prompt TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        logger.info("SQLite database initialized successfully.")

    except sqlite3.Error as e:
        logger.error(f"Error initializing SQLite DB: {e}", exc_info=True)
    finally:
        if conn:
            conn.close()

def init_chromadb():
    """
    Initializes and returns a persistent ChromaDB client.
    
    Returns:
        A tuple of (client, collection) or (None, None) on failure.
    """
    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        # Use a sentence transformer model for creating embeddings
        embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-mpnet-base-v2"  
        )

        # Get or create the collection
        collection = client.get_or_create_collection(
            name="creations",
            embedding_function=embedding_func,
            metadata={"hnsw:space": "cosine"} # Using cosine similarity
        )
        logger.info(f"ChromaDB client initialized. Collection 'creations' is ready.")
        return client, collection
    
    except Exception as e:
        logger.error(f"Error initializing ChromaDB: {e}", exc_info=True)
        return None, None


def save_generation(session_id: str, user_prompt: str, enhanced_prompt: str) -> int:
    """
    Saves a generation record to both SQLite and ChromaDB.

    Args:
        session_id: The ID of the current user session.
        user_prompt: The original prompt from the user.
        enhanced_prompt: The final enhanced prompt used for generation.

    Returns:
        The integer ID of the newly created prompt record, or -1 on failure.
    """
    prompt_id = -1

    # 1. Persist in SQLite
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute(
            "INSERT INTO prompts (session_id, user_prompt, enhanced_prompt) VALUES (?, ?, ?)",
            (session_id, user_prompt, enhanced_prompt)
        )

        prompt_id = c.lastrowid
        conn.commit()
        logger.info(f"Saved prompt ID {prompt_id} to SQLite.")

    except sqlite3.Error as e:
        logger.error(f"Failed to save prompt to SQLite: {e}", exc_info=True)
        return -1
    finally:
        if conn:
            conn.close()

    # 2. Persist embedding in ChromaDB
    if prompt_id != -1 and enhanced_prompt:
        try:
            _, collection = init_chromadb()
            if collection:
                collection.add(
                    ids=[str(prompt_id)],
                    documents=[enhanced_prompt],
                    metadatas=[{"session_id": session_id}]
                )
                logger.info(f"Saved prompt ID {prompt_id} embedding to ChromaDB.")

        except Exception as e:
            logger.error(f"Failed to save embedding to ChromaDB: {e}", exc_info=True)

    return prompt_id

def find_similar_prompts(query_text: str, k: int = 3) -> List[Dict[str, Any]]:
    """
    Finds prompts in ChromaDB that are semantically similar to the query text.

    Args:
        query_text: The text to search for.
        k: The number of similar results to return.

    Returns:
        A list of dictionaries containing id, user_prompt, enhanced_prompt, timestamp, and similarity distance for each result.
    """
    try:
        _, collection = init_chromadb()
        if not collection:
            return []

        results = collection.query(
            query_texts=[query_text],
            n_results=k
        )

        if not results or not results.get("ids"):
            logger.info("No similar prompts found in ChromaDB")
            return []

        retrieved_ids = results["ids"][0]
        distances = results["distances"][0]
        logger.info(f"ChromaDB returned {len(retrieved_ids)} similar prompts with IDs: {retrieved_ids}")

        # Fetch full data from SQLite using the retrieved IDs
        conn = get_db_connection()
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        
        # Create a placeholder string for the IN clause
        placeholders = ','.join('?' for _ in retrieved_ids)
        query = f"SELECT id, user_prompt, enhanced_prompt, timestamp FROM prompts WHERE id IN ({placeholders})"
        
        c.execute(query, retrieved_ids)
        rows = c.fetchall()
        logger.info(f"SQLite returned {len(rows)} matching records")
        
        # Mapping rows to a dictionary for easy lookup
        rows_by_id = {str(row['id']): dict(row) for row in rows}

        # Combine results with distances
        final_results = []
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in rows_by_id:
                record = rows_by_id[doc_id]
                record['distance'] = distances[i]
                final_results.append(record)
                logger.info(f"Memory record {doc_id}: id={record.get('id')}, user_prompt='{record.get('user_prompt', '')[:50]}...', enhanced_prompt='{record.get('enhanced_prompt', '')[:50]}...', timestamp={record.get('timestamp')}, distance={record.get('distance')}")

        logger.info(f"Returning {len(final_results)} similar prompts with fields: id, user_prompt, enhanced_prompt, timestamp, distance")
        
        return final_results
    except Exception as e:
        logger.error(f"Error finding similar prompts: {e}", exc_info=True)
        return []
    finally:
        if 'conn' in locals() and conn:
            conn.close()


# Initialize on import
init_sqlite()