from __future__ import print_function
from pyspark import SparkContext
from pyspark.mllib.feature import HashingTF, IDF
from pyspark.sql import Window, SparkSession
from pyspark.sql.functions import monotonically_increasing_id
import pyspark.sql.functions as F
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
# sc = SparkContext()
#
# # Load documents (one per line).
# documents = sc.textFile("*.txt").map(lambda line: line.split(" "))
#
#
# tf = HashingTF().transform(documents)
# idf = IDF().fit(tf)
# tfidf = idf.transform(tf)
#
#
# print("tf vector output")
# tf_v=tf.first()
# print(tf_v)
# print("tfifd vector output")
# tfidv_v = tfidf.first()
# print(tfidv_v)


#df = sc.parallelize(tfidf).toDF(["id", "tokens"])



# df = tfidf.map(lambda v: (v, )).toDF(["features"])
# df.show(1, False)



# hashingTF = HashingTF()
# tf = hashingTF.transform(documents)
# tf.cache()
#
# idf = IDF(minDocFreq=2).fit(tf)
# tfidf = idf.transform(tf)


if __name__ == "__main__":
    # spark = SparkContext(appName="TFIDFExample")  # SparkContext
    spark = SparkSession.builder.appName("TfIdf-tokenizing").getOrCreate()
    # $example on$
    # Load documents (one per line).
    # documents = sc.textFile("*.txt").map(lambda line: line.split(" "))
    documents = spark.read.text("*.txt")
    documents = documents.withColumn("doc_id", F.row_number().over(Window.orderBy('value')))
    documents.printSchema()

    # creating tokens/words from the sentence data
    tokenizer = Tokenizer(inputCol="value", outputCol="words")
    wordsData = tokenizer.transform(documents)

    # applying tf on the words data
    hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=20)
    featurizedData = hashingTF.transform(wordsData)

    # calculating the IDF
    idf = IDF(inputCol="rawFeatures", outputCol="features")
    idfModel = idf.fit(featurizedData)
    rescaledData = idfModel.transform(featurizedData)

    # displaying the results
    rescaledData.select("doc_id", "features").show(truncate=False)

    # closing the spark session
    spark.stop()

    # hashingTF = HashingTF()
    # tf = hashingTF.transform(documents)
    #
    # # While applying HashingTF only needs a single pass to the data, applying IDF needs two passes:
    # # First to compute the IDF vector and second to scale the term frequencies by IDF.
    # tf.cache()
    # idf = IDF().fit(tf)
    # tfidf = idf.transform(tf)
    #
    # idfIgnore = IDF(minDocFreq=2).fit(tf)
    # tfidfIgnore = idfIgnore.transform(tf)
    #
    # tf_v = tf.first()
    # tfidf_v = tfidf.first()

    # print("tf values output")
    # print(tf_v)
    #
    # print("tf idf values output")
    # print(tfidf_v)


#
#
#
#
#
#
#
#     print("tfidf:")
#     for each in tfidf.collect():
#         print(each)
#
#     print("tfidfIgnore:")
#     for each in tfidfIgnore.collect():
#         print(each)
#
#     sc.stop()
# #
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#













# from __future__ import print_function
# from pyspark.ml.feature import HashingTF, IDF, Tokenizer
# from pyspark.sql import SparkSession
# from pyspark import SparkContext
# from pyspark.sql import SQLContext
# from pyspark.sql.types import *







# creating spark session
#spark = SparkSession.builder.appName("TfIdf Example").getOrCreate()
#
# # Load relevant objects
#val_df = spark.read.option("header", "false").csv("Abstract1.txt")
#print(val_df.collect())

# from pyspark.ml.feature import CountVectorizer
# cv = CountVectorizer(inputCol="words", outputCol="features")
# model = cv.fit(df)result = model.transform(df)
# result.show(truncate=False)

# sc = SparkContext('local')
# sc.setLogLevel("WARN")
# log_txt = sc.textFile("Abstract1.txt")
# sqlContext = SQLContext(sc)
#
# # Construct fields with names from the header, for creating a DataFrame
# header = log_txt.first()
# log_txt = log_txt.filter(lambda line: line != header)
# print(log_txt)
#
#
# temp_var = log_txt.map(lambda k: k.split("\\t"))
#
# log_df=temp_var.toDF(header.split("\\t"))
# log_df.show()

# fields = [StructField(field_name, StringType(), True)
#       for field_name in header.split(',')]



# #setup the same way you have it
# log_txt=sc.textFile("/path/to/data/sample_data.txt")
# header = log_txt.first()
#
# #filter out the header, make sure the rest looks correct
# log_txt = log_txt.filter(lambda line: line != header)
# log_txt.take(10)
#   [u'0\\tdog\\t20160906182001\\tgoogle.com', u'1\\tcat\\t20151231120504\\tamazon.com']
#
# temp_var = log_txt.map(lambda k: k.split("\\t"))
#
# #here's where the changes take place
# #this creates a dataframe using whatever pyspark feels like using (I think string is the default). the header.split is providing the names of the columns
# log_df=temp_var.toDF(header.split("\\t"))
# log_df.show()
#
#
#
# # creating spark session
# spark = SparkSession.builder.appName("TfIdf Example").getOrCreate()
#
# # creating spark dataframe with the input data. You can also read the data from file. label represents the 3 documnets (0.0,0.1,0.2)
# sentenceData = spark.createDataFrame([
#         (0.0, "Welcome to KDM TF_IDF Tutorial."),
#         (0.1, "Learn Spark ml tf_idf in today's lab."),
#         (0.2, "Spark Mllib has TF-IDF.")
#     ], ["label", "sentence"])
#
# #print(type(sentenceData))
# # creating tokens/words from the sentence data
# tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
# wordsData = tokenizer.transform(sentenceData)
#
# # applying tf on the words data
# hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=20)
# featurizedData = hashingTF.transform(wordsData)
# print(featurizedData)
# # alternatively, CountVectorizer can also be used to get term frequency vectors
#
# # calculating the IDF
# idf = IDF(inputCol="rawFeatures", outputCol="features")
# idfModel = idf.fit(featurizedData)
# rescaledData = idfModel.transform(featurizedData)
#
# #displaying the results
# rescaledData.select("label", "sentence").show(truncate=False)
# rescaledData.select("label", "features").show(truncate=False)
#
#
# #closing the spark session
# spark.stop()