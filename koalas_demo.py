import pandas as pd
from databricks import koalas as ks
from pyspark.sql import SparkSession
from pyspark import SparkContext
import time
import logging

#hush Spark chatter
logger = SparkContext.getOrCreate()._jvm.org.apache.log4j
logger.LogManager.getLogger("org"). setLevel( logger.Level.ERROR )
logger.LogManager.getLogger("akka").setLevel( logger.Level.ERROR )

print("Starting Script")

start_time = time.time()
path_to_file = '/usr/local/bin/breast_cancer_data.csv'
# Pandas
df = pd.read_csv(path_to_file)  
# perform expensive operations
df = df.sample(frac=1)

execution_time = time.time() - start_time
print("Dataframe with Pandas:")
print(df)
print(f"Execution time was: {execution_time}")

start_time = time.time()
# Koalas on top of Spark df
df = ks.read_csv(path_to_file)  
# perform expensive operations
df = df.sample(frac=float(1))

print("Spark Dataframe with Koalas:")
print(df)
print(f"Execution time was: {execution_time}")

print("Exiting Script")
