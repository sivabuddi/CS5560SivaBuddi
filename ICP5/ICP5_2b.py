from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.sql import SparkSession, Window
import pyspark.sql.functions as F
from nltk.stem import WordNetLemmatizer
from pyspark.sql.types import ArrayType, StringType
from nltk.stem.snowball import SnowballStemmer
from pyspark.ml.feature import Word2Vec

lemmtizer = WordNetLemmatizer()


def lemmetize(input_list):
    print(input_list)
    if(len(input_list) == 1):
        return list()
    return [lemmtizer.lemmatize(word) for word in input_list]


spark = SparkSession.builder.appName("TfIdf-Lemmetization").getOrCreate()

lemmetize = F.udf(lemmetize)
# spark.udf.register("lemmetize", lemmetize)

documents = spark.read.text("*.txt")
documents = documents.withColumn("doc_id", F.row_number().over(Window.orderBy('value')))

documents.printSchema()
# creating tokens/words from the sentence data
tokenizer = Tokenizer(inputCol="value", outputCol="words")
wordsData = tokenizer.transform(documents)
wordsData.show()

stemmer = SnowballStemmer(language='english')
stemmer_udf = F.udf(lambda tokens: [stemmer.stem(token) for token in tokens], ArrayType(StringType()))
wordsData = wordsData.withColumn("lemms", stemmer_udf("words"))

# Learn a mapping from words to Vectors.
word2Vec = Word2Vec(vectorSize=3, minCount=0, inputCol="lemms", outputCol="result")
model = word2Vec.fit(wordsData)
result = model.transform(wordsData)

# for row in result.collect():
#     text, vector = row
#     # printing the results
#     print("Text: [%s] => \nVector: %s\n" % (", ".join(text), str(vector)))

# showing the synonyms and cosine similarity of the word in input data
synonyms = model.findSynonyms("enough", 1)  # its okay for certain words , real bad for others
synonyms.show(5)

# closing the spark session
spark.stop()