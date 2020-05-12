import pandas as pd
from databricks import koalas as ks
from pyspark.sql import SparkSession
from pyspark import SparkContext
import logging

#hush Spark chatter
logger = SparkContext.getOrCreate()._jvm.org.apache.log4j
logger.LogManager.getLogger("org"). setLevel( logger.Level.ERROR )
logger.LogManager.getLogger("akka").setLevel( logger.Level.ERROR )

print("Starting Script")

# Pandas
df = pd.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]})

# Rename columns
df.columns = ['a', 'b', 'c1']

# Do some operations in place
df['a2'] = df.a * df.a

print("Dataframe with Pandas:")
print(df)


# Koalas on top of Spark df
df = ks.DataFrame({'a': [1, 2], 'b': [3, 4], 'c': [5, 6]})

df.columns = ['a', 'b', 'c1']
df['a2'] = df.a * df.a

print("Spark Dataframe with Koalas:")
print(df)

print("Exiting Script")
