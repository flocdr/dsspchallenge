# import and initialization
import os
os.environ['PYSPARK_SUBMIT_ARGS'] = '--master local[7]  --packages graphframes:graphframes:0.8.2-spark3.1-s_2.12  pyspark-shell'

import findspark
findspark.init()

import pyspark
from pyspark.sql import SparkSession

import graphframes

ss = SparkSession.builder.appName('challenge').getOrCreate()
sc = ss.sparkContext


# loading data
dft = ss.read.csv("node_information.csv")
df_nodes = dft.toDF("id", "year", "topics", "authors", "temp1", "text")
df_nodes.createOrReplaceTempView("nodes")

dft = ss.read.option("delimiter", " ").csv("training_set.txt")
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

    return edges.join(in_degrees, edges.target == in_degrees.id).drop(in_degrees.id)



# FEATURE 2 : Graphe d'auteurs
def explode_authors(row):
    my_input=row.asDict()

    my_output=[]
    #if my_input['author']==None or my_input['cited'] == None : my_output['author']='' #We want o avoid any null issues

    for author in my_input['author'].split(','):
        for cited in my_input['cited'].split(','):
            ret = {}
            ret['src'] = str(author).strip()
            ret['dst'] = str(cited).strip()
            newRow = Row(*ret.keys()) #a. the Row object specification (column names)
            newRow = newRow(*ret.values())#b. the corresponding column values

            my_output.append(newRow)
    return my_output

def get_authors(edges, nodes):
    
    edges_authors_src = edges.join(nodes, edges.source == nodes.id)
    edges_authors_dst = edges_authors_src.join(nodes, edges.target == nodes.id)

    return edges_authors_dst


# PIPELINE

in_degrees = articles_graph.inDegrees
data = in_degrees_feature(df_edges, in_degrees)
# data.show()

data = get_authors(data, df_nodes)
data.show()

