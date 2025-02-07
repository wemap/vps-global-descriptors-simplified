from argparse import ArgumentParser
import numpy as np
from pgvector.psycopg import register_vector
from joblib import load

from db_connection import create_aws_connection, create_local_connection
from logger import logger
from import_data import *
from benchmark_pgvector import *
from benchmark_numpy import *

def parse_arguments():
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--aws", action="store_true", help="Use AWS for pgvector")
    arg_parser.add_argument("--force_import", action="store_true", help="Skip database existence check and reimport data anyway")
    arg_parser.add_argument("--num_nearest_neighbors", type=int, default=20, help="Number of nearest neighbors to retrieve")
    arg_parser.add_argument("--iterations", type=int, default=50, help="Number of iterations for the benchmark")
    return arg_parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    force_import = args.force_import
    num_nearest_neighbors = args.num_nearest_neighbors
    iterations = args.iterations

    logger.info("Benchmarking on " + 
                ("AWS" if args.aws else "local") + 
                " with " + str(num_nearest_neighbors) + " nearest neighbors and " + 
                str(iterations) + " iterations")
    
    random_global_descriptors = [np.random.rand(4096) for _ in range(iterations)]

    #########################################################################################
    #  0.1. Goal
    #  - Find the 10 nearest neighbors of a query descriptor in the database
    #
    #  0.2. Infos
    #  - Descriptor size: 4096 dimensions of float32
    #  - Database size: 62256 descriptors
    #  - Theoretically computed size: 4096 * 4 bytes * 62256 = 972MB
    #
    #  0.3. Limitation of current implementation
    #  - We store dozen (or more) maps on each server, so it does not scale because 
    #   everything is in memory
    #########################################################################################

    #########################################################################################
    #  1. Current implementation 
    #   Limit: does not scale because everything is in memory
    #########################################################################################

    logger.info("= Benchmarking numpy =")
    all_global_descriptors = np.load("data/global_descriptors_4096.npy")
    benchmark_numpy(all_global_descriptors, random_global_descriptors, num_nearest_neighbors)

    #########################################################################################
    #  2. pgvector
    #########################################################################################

    connection = create_aws_connection() if args.aws else create_local_connection()
    with connection.cursor() as cur:
        cur.execute('CREATE EXTENSION IF NOT EXISTS vector')
    register_vector(connection)

    #########################################################################################
    #  2.a. pgvector without indexing
    #########################################################################################
    
    logger.info("= Benchmarking pgvector =")
    import_data_pgvector(all_global_descriptors, connection, force_import)
    benchmark_pgvector(connection, random_global_descriptors, "normal", num_nearest_neighbors)

    #########################################################################################
    #  2.b. pgvector with HNSW binary quantize index
    #########################################################################################

    logger.info("= Benchmarking pgvector with HNSW binary quantize index =")
    import_data_pgvector_hnsw_binary_quantize(all_global_descriptors, connection, force_import)
    benchmark_pgvector(connection, random_global_descriptors, "hnsw_binary_quantize", num_nearest_neighbors)


    #########################################################################################
    #  2.c. pgvector with PCA and indexing
    #   These tests have been done just to see the performance of pgvector indexing
    #   Limitations: 
    #   - does not scale because pca_model is stored in memory
    #   - degraded quality of results
    #########################################################################################

    all_global_descriptors_pca = np.load("data/global_descriptors_2000.npy")
    
    # Transform random global descriptors to PCA space
    pca = load('data/pca_model.joblib')
    random_global_descriptors_pca = []
    for query_global_descriptor in random_global_descriptors:
        # In theory this should be taken into account in the benchmark
        query_global_descriptor_pca = pca.transform(query_global_descriptor.reshape(1, -1)).flatten()
        random_global_descriptors_pca.append(query_global_descriptor_pca)

    logger.info("= Benchmarking pgvector with PCA + IVFFlat =")
    import_data_pgvector_pca_ivfflat(all_global_descriptors_pca, connection, force_import)
    benchmark_pgvector(connection, random_global_descriptors_pca, "ivfflat", num_nearest_neighbors)

    logger.info("= Benchmarking pgvector with PCA + HNSW =")
    import_data_pgvector_pca_hnsw(all_global_descriptors_pca, connection, force_import)
    benchmark_pgvector(connection, random_global_descriptors_pca, "hnsw", num_nearest_neighbors)


    connection.close()