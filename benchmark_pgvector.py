import numpy as np
import time
from logger import logger

def benchmark_pgvector(connection, query_global_descriptors, type="ivfflat" or "hnsw" or "normal", num_nearest_neighbors=20):
    total_time = 0
    for query_global_descriptor in query_global_descriptors:
        start_time = time.time()
        calculate_similarity_pgvector(query_global_descriptor, connection, type, num_nearest_neighbors)
        total_time += time.time() - start_time
    
    avg_time = total_time / len(query_global_descriptors)
    logger.info(f"  - Average time: {avg_time*1000:.0f} ms on {len(query_global_descriptors)} queries")

def calculate_similarity_pgvector(query_global_descriptor, connection, type="ivfflat" or "hnsw" or "normal", num_nearest_neighbors=20):

    table_name = "global_descriptors_pca_ivfflat" if type == "ivfflat" else "global_descriptors_pca_hnsw" if type == "hnsw" else "global_descriptors"

    with connection.cursor() as cur:
        cur.execute(f"SELECT id FROM {table_name} ORDER BY (descriptor <#> %s)*-1 DESC LIMIT %s;", (query_global_descriptor, num_nearest_neighbors))
        results = cur.fetchall()

    return np.array([r[0] for r in results])