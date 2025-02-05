# Benchmarking pgvector vs numpy

## Description

This project aims to benchmark the performance of pgvector against current numpy implementation which is not scalable

## Use case

**Find the 20 nearest neighbors of a query descriptor in the database using scalar product**

- Descriptor size: 4096 dimensions of float32
- Database size: 62256 descriptors
- Theoretically computed size: 4096 * 4 bytes * 62256 = 972MB

## Limitation of current implementation

We store dozen (or more) databases on each server and it does not scale because  everything is in RAM. For example, if we have 20 maps, it means 20 * 972MB ~= 20GB in RAM.

Note: databases are strictly independent from each other.

## How to run the benchmark

```bash
pip install -r requirements.txt
python main.py [options]
```

Options:

- `--aws`: Use AWS Aurora PostgreSQL instead of local PostgreSQL. See [db_connection.py](db_connection.py) for more details.
- `--force_import`: Skip database existence check and reimport data anyway
- `--num_nearest_neighbors`: Number of nearest neighbors to retrieve
- `--iterations`: Number of iterations for the benchmark


## Results

### Local PostgreSQL

```bash
Benchmarking on local with 20 nearest neighbors and 50 iterations
= Benchmarking numpy =
  - Average time: 119 ms on 50 queries
= Benchmarking pgvector =
  - Import in: 23608 ms
  - Average time: 428 ms on 50 queries
= Benchmarking pgvector with PCA + IVFFlat =
  - Import in: 26140 ms
  - Average time: 260 ms on 50 queries
= Benchmarking pgvector with PCA + HNSW =
  - Import in: 270990 ms
  - Average time: 225 ms on 50 queries
```

### AWS Aurora PostgreSQL

TODO

