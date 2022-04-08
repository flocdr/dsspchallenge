from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.functions import lower, col
import numpy as np
from pyspark.ml.linalg import SparseVector, VectorUDT
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from pyspark.sql.types import ArrayType


spark = SparkSession.builder.appName('example').getOrCreate()
sc = spark.sparkContext

df_articles = spark.read.csv("./node_information.csv",sep=",").toDF("ID","Year","Title","Authors","Empty","Abstract")

#Classic TF-IDF (with hashing)
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
#1. split text field into words (Tokenize)
tokenizer = Tokenizer(inputCol="Title", outputCol="words_title")
df_articles = tokenizer.transform(df_articles)
print("New column with tokenized text")
print(df_articles.head(1))
print("################")

#How do I save the result:
# df_articles.write.option("compression","gzip").parquet("tf_idf_example.parquet")



#2. compute term frequencies
hashingTF = HashingTF(inputCol="words_title", outputCol="tf_title")
df_articles = hashingTF.transform(df_articles)
print("TERM frequencies:")
print(df_articles.head(1))
print("################")


#3. IDF computation
idf = IDF(inputCol="tf_title", outputCol="tf_idf_title")
idfModel = idf.fit(df_articles) #model that contains "dictionary" and IDF values
df_articles = idfModel.transform(df_articles)
print("TF_IDF vector:")
print(df_articles.head(1))
print("################")