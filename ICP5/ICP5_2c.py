from pyspark.ml.feature import Tokenizer, NGram
from pyspark.sql import SparkSession, Window
import pyspark.sql.functions as F
from pyspark.ml.feature import Word2Vec

spark = SparkSession.builder.appName("TfIdf-Ngram").getOrCreate()
documents = spark.read.text("*.txt")
documents = documents.withColumn("doc_id", F.row_number().over(Window.orderBy('value')))

documents.printSchema()
# creating tokens/words from the sentence data
tokenizer = Tokenizer(inputCol="value", outputCol="words")
wordsData = tokenizer.transform(documents)

ngram = NGram(n=2, inputCol="words", outputCol="ngrams")
wordsData = ngram.transform(wordsData)

# Learn a mapping from words to Vectors.
word2Vec = Word2Vec(vectorSize=3, minCount=0, inputCol="ngrams", outputCol="result")
model = word2Vec.fit(wordsData)
result = model.transform(wordsData)


# showing the synonyms and cosine similarity of the word in input data
synonyms = model.findSynonyms("separation", 1)  # its okay for certain words , real bad for others
synonyms.show(5)

# closing the spark session
spark.stop()

# for row in result.collect():
#     text, vector = row
#     # printing the results
#     print("Text: [%s] => \nVector: %s\n" % (", ".join(text), str(vector)))

