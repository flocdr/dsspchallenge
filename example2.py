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

df_articles = spark.read.csv("/dssp/datacamp/articles_text.csv",sep=",").toDF("Title","Intro")
df_links = spark.read.csv("/dssp/datacamp/edges_class.csv",sep=",").toDF("source","target","class")
df_articles = df_articles.withColumn("title_lower",lower(col("Title")))

df_articles.createOrReplaceTempView("text")
df_links.createOrReplaceTempView("links")

inner_join=spark.sql("""Select source,target, text1.Intro as txt1,class 
								from links 
								inner join text as text1 on text1.title_lower==source""")
								
#create features with word count on source								
def transform(row):
	data=row.asDict()#Rows are immutable; We convert them to dictionaries
	if data['source']==None : data['source']='' #We want o avoid any null issues
	#get unique words
	text1=set([x.lower() for x in  data['source'].encode('utf-8').split()])
	#create a new column with an array of number to be the features
	data["features"]=[len(text1)]#only one feature
	#keep only features and class
	ret={}
	ret["features"]=data["features"]
	ret["class"]=float(data["class"])
	#convert the dictionary back to a Row
	newRow = Row(*ret.keys()) #a. the Row object specification (column names)
	newRow = newRow(*ret.values())#b. the corresponding column values
	return newRow

data=inner_join.rdd.map(transform).toDF()	
print("New dataframe with map on RDD")
print(data.head(1))
print("################")	
#models use sparse arrays	

#Lets apply a function directly on the dataframe
#UDF
#applying functions to a DATAFRAME requires the "SQL" logic of 
#User Defined Functions (UDF)
#as an example: convert features to sparse array
#1. define what data type your UDF returns VectorUDT : UDF SCHEMA
custom_udf_schema = VectorUDT()
#2. define function 
def to_sparse_(v):
		import numpy as np
		if isinstance(v, SparseVector):
			return v
		vs = np.array(v)
		nonzero = np.nonzero(vs)[0]
		return SparseVector(len(v), nonzero, vs[nonzero])
#3. create a udf from that function and the schema
to_sparse = udf(to_sparse_,custom_udf_schema)
#4. apply UDF to DF and create new column

data= data.withColumn('feats',to_sparse(data.features))

print("new sparse array column")
print(data.head(1))
print("################")


#Build a model on our data

from pyspark.ml.classification import LogisticRegression
#simple split
(train,test)=data.rdd.randomSplit([0.8,0.2])
#we only have 10 iterations for this example
#pay attention to the input/output columns
logistic=LogisticRegression(featuresCol="feats",labelCol="class",predictionCol='class_pred',rawPredictionCol="class_pred_raw",maxIter=10)
lrModel = logistic.fit(train.toDF())
result=lrModel.transform(test.toDF())
print("prediction output")
print(result.head(1))
print("################")

#perform evaluation

from pyspark.ml.evaluation import BinaryClassificationEvaluator
evaluator = BinaryClassificationEvaluator(rawPredictionCol="class_pred_raw",labelCol='class',metricName="areaUnderROC",)
print("evaluation")
print(evaluator.evaluate(result))
print("################")