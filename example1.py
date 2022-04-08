from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.functions import lower, col

#start "spark session" 
spark = SparkSession.builder.appName('example').getOrCreate()
sc = spark.sparkContext

#Load articles
df_articles = spark.read.csv("/dssp/datacamp/articles_text.csv",sep=",").toDF("Title","Intro")
#print one Row from articles
print("articles dataframe")
print(df_articles.head(1))
print("################")

#Load edges
df_links = spark.read.csv("/dssp/datacamp/edges_class.csv",sep=",").toDF("source","target","class")
print("links dataframe")
print(df_links.head(1))
print("################")

#Apply function on Title and create new column
df_articles = df_articles.withColumn("title_lower",lower(col("Title")))
print("create new column")
print(df_articles.head(1))
print("################")

#do a join 
inner_join_example = df_articles.join(df_links, df_articles.title_lower == df_links.source)
print("join with python function")
print(inner_join_example.head(1))
print("################")

#drop a column
inner_join_example = inner_join_example.drop(inner_join_example.Title)
print("drop a column")
print(inner_join_example.head(1))
print("################")

#We can do both and more complex stuff with SQL as well

#But first we need tables to refer to 

df_articles.createOrReplaceTempView("text")
df_links.createOrReplaceTempView("links")
print("simple select with SQL")
print(spark.sql("Select source,target,class from links where source like '%par%'" ).head(1))
print("################")

#Count how many elements per class
print("count classes")
print(spark.sql("Select class,count(*) from links group by class" ).show())
print("################")

#Join and select some of the columns
inner_join_example2=spark.sql("""Select source,target, text1.Intro as txt1,class 
								from links 
								inner join text as text1 on text1.title_lower==source""")
print("join and column selection with SQL")
print(inner_join_example2.head(1))
print("################")