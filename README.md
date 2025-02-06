# Benchmarking pgvector vs numpy

This project aims to benchmark the performance of **pgvector** against current **numpy** implementation which is not scalable. 

## Use case

**Find the 20 nearest neighbors of a query descriptor in the database using scalar product**

- Descriptor size: 4096 dimensions of float32
- Database size: 62256 descriptors
- Theoretically computed size: 4096 * 4 bytes * 62256 = 972MB

A reasonnable time of execution is less than 200 ms.

## Limitation of current implementation

We store dozen (or more) databases on each server and it does not scale because  everything is in RAM. For example, if we have 20 maps, it means 20 * 972MB ~= 20GB in RAM.

Note: databases are strictly independent from each other.

## How to run the benchmark

Download the data from [here](https://drive.google.com/file/d/1L1ASpuUwxriy0pvH7BNsTckvnqdFTriW/view?usp=drive_link), unzip it and put it in the `data` folder at the root of the project.


Then, install the dependencies and run the benchmark with:
```bash
pip install -r requirements.txt
python main.py [options]
```

Options:

- `--aws`: Use AWS Aurora PostgreSQL instead of local PostgreSQL. See [db_connection.py](db_connection.py) for more details about credentials.
- `--force_import`: Skip database existence check and reimport data anyway.
- `--num_nearest_neighbors`: Number of nearest neighbors to retrieve.
- `--iterations`: Number of iterations for the benchmark.

In order to run it *locally*, you can mount a PostgreSQL server via docker using `cd postgres-server && docker compose up`.

## Results

### Local PostgreSQL

- **Server:** Apple M4 Pro via `cd postgres-server && docker compose up`
- **Client:** Apple M4 Pro

```bash
Benchmarking on local with 20 nearest neighbors and 50 iterations
= Benchmarking numpy =
  - Average request time: 119 ms
= Benchmarking pgvector =
  - Import in: 23608 ms
  - Average request time: 428 ms
= Benchmarking pgvector with PCA + IVFFlat =
  - Import in: 26140 ms
  - Average request time: 260 ms
= Benchmarking pgvector with PCA + HNSW =
  - Import in: 270990 ms
  - Average request time: 225 ms
```

### AWS Aurora PostgreSQL + g5 client

- **Server:** db.r7g.large with `default.aurora-postgresql15` (~$300/month) 
- **Client:** g5.xlarge

```bash
Benchmarking on AWS with 20 nearest neighbors and 50 iterations
= Benchmarking numpy =
  - Average request time: 200 ms
= Benchmarking pgvector =
  - Import in: 13952 ms
  - Average request time: 583 ms
= Benchmarking pgvector with PCA + IVFFlat =
  - Import in: 21042 ms
  - Average request time: 426 ms
= Benchmarking pgvector with PCA + HNSW =
  - Import in: 571858 ms
  - Average request time: 424 ms
```

## More info

- We did not try to use half precision for the descriptor yet.
- PCA 4096 -> 2000 dimensions degrades the quality of the results too much.
- We can accept recall higher than 90%.