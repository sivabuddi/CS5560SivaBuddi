from pyspark.ml.feature import HashingTF, IDF, Tokenizer, NGram
from pyspark.sql import SparkSession, Window
import pyspark.sql.functions as F
# creating spark session
if __name__ == "__main__":
    # spark = SparkContext(appName="TFIDFExample")  # SparkContext
    spark = SparkSession.builder.appName("TfIdf-tokenizing").getOrCreate()

 # documents = sc.textFile("*.txt").map(lambda line: line.split(" "))
documents = spark.read.text("*.txt")
documents = documents.withColumn("doc_id", F.row_number().over(Window.orderBy('value')))

documents.printSchema()
# creating tokens/words from the sentence data
tokenizer = Tokenizer(inputCol="value", outputCol="words")
wordsData = tokenizer.transform(documents)

ngram = NGram(n=2, inputCol="words", outputCol="ngrams")
ngramDataFrame = ngram.transform(wordsData)

# applying tf on the words data
hashingTF = HashingTF(inputCol="ngrams", outputCol="rawFeatures", numFeatures=20)
featurizedData = hashingTF.transform(ngramDataFrame)
# alternatively, CountVectorizer can also be used to get term frequency vectors

# calculating the IDF
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

# displaying the results
rescaledData.select("doc_id", "features").show(truncate=False)
# closing the spark session
spark.stop()