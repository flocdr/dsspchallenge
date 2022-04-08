from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.functions import lower, col




spark = SparkSession.builder.appName('example').getOrCreate()
sc = spark.sparkContext

df_links = spark.read.csv("/dssp/datacamp/edges_class.csv",sep=",").toDF("source","target","class")


from graphframes import *

#to build a graph we need vertices and edges
# source, targer, class -> lambda  -> [[source],[target]] -> flatMap -> 
#[target]
#[source]
v=df_links.rdd.flatMap(lambda x:[[x['source']],[x['target']]]).toDF(["id"])
e=df_links.select(col("source").alias("src"),col("target").alias("dst"))

g = GraphFrame(v, e)
print("edges part of graph")
print(g.edges.show())
print("################")
df_indeg=g.inDegrees
print("in degrees")
print(df_indeg.show())
print("################")
df_outdeg=g.outDegrees
print("out degrees")
print(df_outdeg.show())
print("################")

