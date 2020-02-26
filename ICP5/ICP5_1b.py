from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.sql import SparkSession, Window
import pyspark.sql.functions as F
from nltk.stem import WordNetLemmatizer
from pyspark.sql.types import ArrayType, StringType
from nltk.stem.snowball import SnowballStemmer

if __name__ == "__main__":
    # spark = SparkContext(appName="TFIDFExample")  # SparkContext
    spark = SparkSession.builder.appName("TfIdf-tokenizing").getOrCreate()

def lemmetize(input_list):
    print(input_list)
    if(len(input_list) == 1):
        return list()
    return [lemmtizer.lemmatize(word) for word in input_list]


lemmtizer = WordNetLemmatizer()
lemmetize = F.udf(lemmetize)

# documents = sc.textFile("*.txt").map(lambda line: line.split(" "))
documents = spark.read.text("*.txt")
documents = documents.withColumn("doc_id", F.row_number().over(Window.orderBy('value')))
documents.printSchema()

# creating tokens/words from the sentence data
tokenizer = Tokenizer(inputCol="value", outputCol="words")
wordsData = tokenizer.transform(documents)
wordsData.show(truncate=False)

stemmer = SnowballStemmer(language='english')
stemmer_udf = F.udf(lambda tokens: [stemmer.stem(token) for token in tokens], ArrayType(StringType()))
wordsData = wordsData.withColumn("lemms", stemmer_udf("words"))
#wordsData = wordsData.withColumn("lemms", lemmetize("words"))
wordsData.show(truncate=False)


# applying tf on the words data
hashingTF = HashingTF(inputCol="lemms", outputCol="rawFeatures", numFeatures=20)
featurizedData = hashingTF.transform(wordsData)
# alternatively, CountVectorizer can also be used to get term frequency vectors

# calculating the IDF
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

# displaying the results
rescaledData.select("doc_id", "features").show(truncate=False)
# closing the spark session
spark.stop()
