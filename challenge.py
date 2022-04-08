# import and initialization
import os

from numpy import full
os.environ['PYSPARK_SUBMIT_ARGS'] = '--master local[7]  --packages graphframes:graphframes:0.8.2-spark3.1-s_2.12  pyspark-shell'

import findspark
findspark.init()

import pyspark
from pyspark.sql import SparkSession

import graphframes

ss = SparkSession.builder.appName('challenge').getOrCreate()
sc = ss.sparkContext


# loading data
dft = ss.read.csv("data/node_information.csv")
df_nodes = dft.toDF("id", "year", "topics", "authors", "temp1", "text")
df_nodes.createOrReplaceTempView("nodes")

dft = ss.read.option("delimiter", " ").csv("data/training_set.txt")
df_edges = dft.toDF("source", "target", "cites")
df_edges.createOrReplaceTempView("edges") 


# transformation des dataframes pour qu'ils soient utilisables par GraphFrames
from pyspark.sql.functions import lower, col

v=df_edges.rdd.flatMap(lambda x:[[x['source']],[x['target']]]).toDF(["id"]).distinct()
e=df_edges.select( col("source").alias("src"),col("target").alias("dst"))


# création du graphe des articles
articles_graph = graphframes.GraphFrame(v ,e)
# articles_graph.vertices.show()


# FEATURE #1 : degrés entrants

def in_degrees_feature(edges, in_degrees):

    return edges.join(in_degrees, edges.dst == in_degrees.id).drop(in_degrees.id)



# AUTHORS GRAPH
def explode_authors(row):
    my_input=row.asDict()

    my_output=[]
    #if my_input['author']==None or my_input['cited'] == None : my_output['author']='' #We want o avoid any null issues

    for author in my_input['author'].split(','):
        for cited in my_input['cited'].split(','):
            ret = {}
            ret['src'] = str(author).strip()
            ret['dst'] = str(cited).strip()
            newRow = pyspark.Row(*ret.keys()) #a. the Row object specification (column names)
            newRow = newRow(*ret.values())#b. the corresponding column values

            my_output.append(newRow)
    return my_output


def get_authors_graph(edges):

    author_edges = ss.sql("select A.authors author, B.authors cited, edges.cites \
        from edges inner join nodes A on edges.target = A.id \
                    inner join nodes B on edges.source = B.id \
        where edges.cites = 1 and A.authors is not null and B.authors is not null")

    author_edges = author_edges.rdd.flatMap(explode_authors).toDF()
    author_edges.createOrReplaceTempView("author_edges")

    author_edges = ss.sql("select src, dst, count(*) as nb from author_edges group by src, dst")

    return author_edges



def get_full_data():

    full_data = ss.sql(
        """
        SELECT edges.source src, edges.target dst, A.year year_src, B.year year_dst, 
            A.topics topics_src, B.topics topics_dst, A.authors authors_src, B.authors authors_dst, A.text text_src, B.text text_dst 
        FROM edges INNER JOIN nodes A ON edges.source = A.id INNER JOIN nodes B ON edges.target=B.id
        WHERE edges.cites = 1 AND A.authors is not null AND B.authors is not null"""
    )

    return full_data



# PIPELINE

in_degrees = articles_graph.inDegrees

data = get_full_data()
data = in_degrees_feature(data, in_degrees)

data.show()
