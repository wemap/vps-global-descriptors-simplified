import time
from logger import logger

def is_table_exists(table_name, connection):
    with connection.cursor() as cur:
        query = "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = %s)"
        cur.execute(query, (table_name,))
        exists = cur.fetchone()[0]
        return exists

def import_data_pgvector(all_global_descriptors, connection, force=False):

    if is_table_exists("global_descriptors", connection) and not force:
        logger.info("  - Skipping import because table already exists")
        return

    start_time = time.time()
    with connection.cursor() as cur:
        cur.execute('DROP TABLE IF EXISTS "global_descriptors"')
        cur.execute('''
            CREATE TABLE "global_descriptors" (
                id int PRIMARY KEY,
                descriptor vector(4096)
            );
            ''')
        cur.executemany('INSERT INTO "global_descriptors" (id, descriptor) VALUES (%s, %s)', enumerate(all_global_descriptors))
    connection.commit()
    logger.info(f"  - Import in: {(time.time() - start_time)*1000:.0f} ms")


def import_data_pgvector_pca_ivfflat(all_global_descriptors_pca, connection, force=False):
    if is_table_exists("global_descriptors_pca_ivfflat", connection) and not force:
        logger.info("  - Skipping import because table already exists")
        return
    
    start_time = time.time()
    with connection.cursor() as cur:
        cur.execute('DROP TABLE IF EXISTS "global_descriptors_pca_ivfflat"')
        cur.execute('''
            CREATE TABLE "global_descriptors_pca_ivfflat" (
                id int PRIMARY KEY,
                descriptor vector(2000)
            );
            ''')
        cur.execute('CREATE INDEX ON "global_descriptors_pca_ivfflat" USING ivfflat (descriptor vector_cosine_ops)')
        cur.executemany('INSERT INTO "global_descriptors_pca_ivfflat" (id, descriptor) VALUES (%s, %s)', enumerate(all_global_descriptors_pca))
    connection.commit()
    logger.info(f"  - Import in: {(time.time() - start_time)*1000:.0f} ms")

def import_data_pgvector_pca_hnsw(all_global_descriptors_pca, connection, force=False):
    if is_table_exists("global_descriptors_pca_hnsw", connection) and not force:
        logger.info("  - Skipping import because table already exists")
        return

    start_time = time.time()
    with connection.cursor() as cur:
        cur.execute('DROP TABLE IF EXISTS "global_descriptors_pca_hnsw"')
        cur.execute('''
            CREATE TABLE "global_descriptors_pca_hnsw" (
                id int PRIMARY KEY,
                descriptor vector(2000)
            );
            ''')
        cur.execute('CREATE INDEX ON "global_descriptors_pca_hnsw" USING hnsw (descriptor vector_cosine_ops)')
        cur.executemany('INSERT INTO "global_descriptors_pca_hnsw" (id, descriptor) VALUES (%s, %s)', enumerate(all_global_descriptors_pca))
    connection.commit()
    logger.info(f"  - Import in: {(time.time() - start_time)*1000:.0f} ms")
