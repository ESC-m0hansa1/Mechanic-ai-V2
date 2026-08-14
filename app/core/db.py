import psycopg                              # the Postgres driver
from pgvector.psycopg import register_vector  # adapter for the vector(384) type
from app.core.config import settings        # our typed config (has database_url)


def get_connection() -> psycopg.Connection:
    """Open a Postgres connection that understands pgvector's `vector` type."""
    conn = psycopg.connect(settings.database_url)  # connect using the URL from .env
    register_vector(conn)                          # register so we can send/recv vectors
    return conn


if __name__ == "__main__":
    # `with ... as` = a context manager: auto-commits on success, closes at the end.
    with get_connection() as conn:
        with conn.cursor() as cur:          # a cursor executes SQL and holds results
            cur.execute("SELECT count(*) FROM chunks;")
            print("chunks in DB:", cur.fetchone()[0])  # fetchone() -> one result row