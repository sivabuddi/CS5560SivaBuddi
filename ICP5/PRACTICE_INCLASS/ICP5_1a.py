from __future__ import print_function
from pyspark.ml.feature import HashingTF, IDF, Tokenizer
from pyspark.sql import SparkSession
import numpy as np



# creating spark session
spark = SparkSession.builder.appName("TfIdf Example").getOrCreate()

# creating spark dataframe with the input data. You can also read the data from file. label represents the 3 documnets (0.0,0.1,0.2)
sentenceData = spark.createDataFrame([
        (0.0, "Welcome to KDM TF_IDF Tutorial."),
        (0.1, "Learn Spark ml tf_idf in today's lab."),
        (0.2, "Spark Mllib has TF-IDF.")
    ], ["label", "sentence"])

# creating tokens/words from the sentence data
tokenizer = Tokenizer(inputCol="sentence", outputCol="words")
wordsData = tokenizer.transform(sentenceData)

# applying tf on the words data
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=20)
featurizedData = hashingTF.transform(wordsData)
print(featurizedData)
# alternatively, CountVectorizer can also be used to get term frequency vectors

# calculating the IDF
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

#displaying the results
np.transpose(rescaledData.select("label", "sentence").show())
rescaledData.select("label", "features").show()


#closing the spark session
spark.stop()