import psycopg

def create_aws_connection():
    return psycopg.connect("TO REPLACE")

def create_local_connection():
    return psycopg.connect("dbname=example_db user=postgres password=password host=localhost port=5432")