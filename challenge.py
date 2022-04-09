# -*- coding: utf-8 -*-
# import and initialization
import numpy as np
# import os

# os.environ['PYSPARK_SUBMIT_ARGS'] = '--master local[7]  --packages graphframes:graphframes:0.8.2-spark3.1-s_2.12  pyspark-shell'

# import findspark
# findspark.init()

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import split, explode, col, lit, concat_ws
from pyspark.sql.types import FloatType

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


# creation du graphe des articles
articles_graph = graphframes.GraphFrame(v ,e)
# articles_graph.vertices.show()


# FEATURE #1 : degres entrants

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



def get_authors_data():

    full_data = ss.sql(
        """
        SELECT edges.source src, edges.target dst,
            A.authors authors_src_, B.authors authors_dst_
        FROM edges INNER JOIN nodes A ON edges.source = A.id INNER JOIN nodes B ON edges.target=B.id
        WHERE edges.cites = 1 AND A.authors is not null AND B.authors is not null"""
    )

    return full_data

def make_authors_pairs(data):
    
    exploded_data_src = data.select(
        data.src,
        data.dst,
        explode(split(data.authors_src_, ",")).alias("author_src"),
        data.authors_dst_,
    )

    exploded_data_dst = exploded_data_src.select(
        exploded_data_src.src,
        exploded_data_src.dst,
        exploded_data_src.author_src,
        explode(split(exploded_data_src.authors_dst_, ",")).alias("author_dst"),
    )

    return exploded_data_dst

# def get_text_data(data):
    # return data.join

#TD-IDF COMPUTATION
def compute_tfidf(df_nodes):
    #1. split text field into words (Tokenize)
    tokenizer = Tokenizer(inputCol="topics", outputCol="words_topics")
    df_nodes = tokenizer.transform(df_nodes)

    #2. compute term frequencies
    hashingTF = HashingTF(inputCol="words_topics", outputCol="tf_topics")
    df_nodes = hashingTF.transform(df_nodes)


    #3. IDF computation
    idf = IDF(inputCol="tf_topics", outputCol="tf_idf_topics")
    idfModel = idf.fit(df_nodes) #model that contains "dictionary" and IDF values
    df_nodes = idfModel.transform(df_nodes)

    return df_nodes

# COSINE SIMILARITY
def cos_sim(vec1, vec2):
    if (np.linalg.norm(vec1) * np.linalg.norm(vec2)) !=0:
        dot_value = np.dot(vec1, vec2) / (np.linalg.norm(vec1)*np.linalg.norm(vec2))
        return dot_value.tolist()

cos_sim_udf = udf(cos_sim, FloatType())




# PIPELINE

in_degrees = articles_graph.inDegrees

data = get_authors_data()
data = make_authors_pairs(data)

# at this point, we have a dataframe with pairs of authors
# we have to make a join to get back the weight of their respectives edges and the group by to get the sum

# data = join_and_group_by(data)

# data = in_degrees_feature(data, in_degrees)
# data = get_text_data(data)

# le reste des features + un beau pipeline tout neuf

data.show()
