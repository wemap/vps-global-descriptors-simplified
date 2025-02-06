import numpy as np
import time
from logger import logger

def benchmark_numpy(all_global_descriptors, global_descriptors_to_query, num_nearest_neighbors=20):
    total_time = 0
    for query_global_descriptor in global_descriptors_to_query:
        start_time = time.time()
        calculate_similarity_numpy(query_global_descriptor, all_global_descriptors, num_nearest_neighbors)
        total_time += time.time() - start_time
    
    avg_time = total_time / len(global_descriptors_to_query)
    logger.info(f"  - Average request time: {avg_time*1000:.0f} ms")
    
def calculate_similarity_numpy(query_global_descriptor, all_global_descriptors, num_nearest_neighbors=20):

    # Calculate similarity scores
    scores = np.einsum('d,jd->j', query_global_descriptor, all_global_descriptors)
    # scores = np.linalg.norm(query_global_descriptor - all_global_descriptors, axis=1)

    # Get top k results
    top_indices = np.argsort(scores)[-num_nearest_neighbors:][::-1]

    # results = list(zip(top_indices, scores[top_indices]))
    return top_indices
