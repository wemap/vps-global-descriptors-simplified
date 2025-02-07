import numpy as np
import time
from logger import logger
from typing import Literal

SetupType = Literal["ivfflat", "hnsw", "hnsw_binary_quantize", "normal"]

def get_table_name(type: SetupType):
    if type == "ivfflat":
        return "global_descriptors_pca_ivfflat"
    elif type == "hnsw":
        return "global_descriptors_pca_hnsw" 
    elif type == "hnsw_binary_quantize":
        return "global_descriptors_hnsw_binary_quantize"
    else:
        return "global_descriptors"

def benchmark_pgvector(connection, query_global_descriptors, type: SetupType = "normal", num_nearest_neighbors=20):
    with connection.cursor() as cur:
        cur.execute(f"SET hnsw.ef_search = 1000;")

    times = []
    for query_global_descriptor in query_global_descriptors:
        start_time = time.time()
        calculate_similarity_pgvector(query_global_descriptor, connection, type, num_nearest_neighbors)
        times.append(time.time() - start_time)
    
    min_time = min(times)
    max_time = max(times)
    avg_time = sum(times) / len(times)
    logger.info(f"  - Request times: min={min_time*1000:.0f}ms, max={max_time*1000:.0f}ms, avg={avg_time*1000:.0f}ms")

def calculate_similarity_pgvector(query_global_descriptor, connection, type: SetupType = "normal", num_nearest_neighbors=20):

    table_name = get_table_name(type)

    with connection.cursor() as cur:
        if type == "hnsw_binary_quantize":
            query = f"""SELECT i.id, i.distance FROM (
                SELECT id, (descriptor <#> %s)*-1 AS distance
                FROM {table_name}
                ORDER BY
                    binary_quantize(descriptor)::bit(4096) <~> binary_quantize(%s)
                LIMIT 1000
            ) as i
            ORDER BY distance DESC
            LIMIT %s;"""
            cur.execute(query, (query_global_descriptor, query_global_descriptor, num_nearest_neighbors))
        else:
            query = f"""SELECT id, (descriptor <#> %s)*-1 AS distance
                FROM {table_name}
                ORDER BY distance DESC
                LIMIT %s;"""
            cur.execute(query, (query_global_descriptor, num_nearest_neighbors))
        results = cur.fetchall()
        # print(results)

    return np.array([r[0] for r in results])