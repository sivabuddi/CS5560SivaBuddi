from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import monotonically_increasing_id
import pyspark.sql.functions as F
from pyspark.ml.feature import Word2Vec

spark = SparkSession.builder.appName("WordToVec-Without NLP").getOrCreate()
documents = spark.read.text("*.txt")
# documents = documents.withColumn("doc_id", monotonically_increasing_id())
documents = documents.withColumn("doc_id", F.row_number().over(Window.orderBy('value')))

documents.printSchema()
# creating tokens/words from the sentence data
tokenizer = Tokenizer(inputCol="value", outputCol="words")
wordsData = tokenizer.transform(documents)

# Learn a mapping from words to Vectors.
word2Vec = Word2Vec(vectorSize=3, minCount=0, inputCol="words", outputCol="result")
model = word2Vec.fit(wordsData)
result = model.transform(wordsData)


# showing the synonyms and cosine similarity of the word in input data
synonyms = model.findSynonyms("response", 2)  # its okay for certain words , real bad for others
synonyms.show(5)

# closing the spark session
spark.stop()