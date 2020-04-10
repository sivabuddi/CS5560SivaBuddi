import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
nltk.download("stopwords")

def extract_features(word_list):
    return dict([(word, True) for word in word_list])



# Create a list of movie review document
documents = []

for category in movie_reviews.categories():
   for fileid in movie_reviews.fileids(category):
      # documents.append((list(movie_reviews.words(fileid)), category))
      documents.append((movie_reviews.words(fileid), category))

if __name__=='__main__':
   # Load positive and negative reviews
   positive_fileids = movie_reviews.fileids('pos')
   negative_fileids = movie_reviews.fileids('neg')
   # print("No.of.postive fileds",positive_fileids)
   # print("No.of.Negative fields",negative_fileids)

   # Total reviews
   print("Total No.of.Reviews in Movies",len(movie_reviews.fileids()))  # Output: 2000

   # Review categories
   print("Categoriacal variables",movie_reviews.categories())  # Output: [u'neg', u'pos']

   # Total positive reviews
   print("Total No.of. positive reviews",len(movie_reviews.fileids('pos')))  # Output: 1000

   # Total negative reviews
   print("Totoal No.of. negative reviews",len(movie_reviews.fileids('neg')))  # Output: 1000

   print("------------------------------------------------------------------------------------------------------------------------------------")
   # Inspect the first positve review
   #print(movie_reviews.raw(fileids=positive_fileids[0]))

   # Fetch all words from the movie reviews corpus
   all_words = [word.lower() for word in movie_reviews.words()]
   print("Total no.of.words in movie corpus",len(all_words))

   # Frequency distribution of words
   from nltk import FreqDist

   all_words_frequency = FreqDist(all_words)

   print("--------------------------------Before data cleaning word frequency----------------------------------")
   print(all_words_frequency)

   # print 10 most frequently occurring words
   print("Top 100 most frequently occuring words before data cleaning")
   print(all_words_frequency.most_common(100))

   # Remove stopwords
   from nltk.corpus import stopwords

   stopwords_english = stopwords.words('english')

   '''
   print(stopwords_english)
   # create a new list of words by removing stopwords from all_words
   all_words_without_stopwords = [word for word in all_words if word not in stopwords_english]

   # print the first 10 words
   print(all_words_without_stopwords[:10])

   # Remove punctuations
   import string
   print("--------------------------------Removing punctuation marks-------------------")
   # print(string.punctuation)
   # print("--------------------------------------------------------------------")

   # create a new list of words by removing punctuation from all_words
   all_words_without_punctuation = [word for word in all_words if word not in string.punctuation]

   # print the first 10 words
   print(all_words_without_punctuation[:10])

   # Remove both stop words and puntuations at a time
   # Let's name the new list as all_words_clean
   # because we clean stopwords and punctuations from the word list
   '''

   import  string
   all_words_clean = []
   for word in all_words:
      if word not in stopwords_english and word not in string.punctuation:
         all_words_clean.append(word)

   #print(all_words_clean[:10])

   # Frequency distribution of cleaned words

   all_words_frequency = FreqDist(all_words_clean)

   print("--------------------------------After data cleaning word frequency----------------------------------")
   print(all_words_frequency)

   # print 10 most frequently occurring words
   print("Top 100 most frequently occuring words after data cleaning")
   print(all_words_frequency.most_common(100))



   # get 2000 frequently occuring words
   most_common_words = all_words_frequency.most_common(2000)

   word_features = [item[0] for item in most_common_words]
   print(word_features[:100]) # get the list instead of tuple

   # Create Feature Set

   def document_features(document):
      # "set" function will remove repeated/duplicate tokens in the given list
      document_words = set(document)
      features = {}
      for word in word_features:
         features['contains(%s)' % word] = (word in document_words)
      return features


   # get the first negative movie review file
   movie_review_file = movie_reviews.fileids('neg')[0]
   print(movie_review_file)


   # print total 10 tuple of the documents list
   print("------------------------------First 10 negative reviews---------------------------------")
   for i in range(10):
      print( documents[i])

   print("------------------------------First 10 postive reviews----------------------------------")
   for i in range(1000,1009):
      print(documents[i])





   feature_set = []
   for (doc, category) in documents:
      feature_set.append((document_features(doc), category))

   print(feature_set[0])


   features_positive = [(extract_features(movie_reviews.words(fileids=[f])),
                         'Positive') for f in positive_fileids]
   features_negative = [(extract_features(movie_reviews.words(fileids=[f])),
                         'Negative') for f in negative_fileids]
   # Split the data into train and test (80/20)
   threshold_factor = 0.8
   threshold_positive = int(threshold_factor * len(features_positive))
   threshold_negative = int(threshold_factor * len(features_negative))

   features_train = features_positive[:threshold_positive] + features_negative[:threshold_negative]
   features_test = features_positive[threshold_positive:] + features_negative[threshold_negative:]
   print("\nNumber of training datapoints:", len(features_train))
   print("Number of test datapoints:", len(features_test))



   # Train a Naive Bayes classifier
   classifier = NaiveBayesClassifier.train(features_train)
   print("\nAccuracy of the classifier:", nltk.classify.util.accuracy(classifier, features_test))

   # print("\nTop 10 most informative words:")
   print(classifier.show_most_informative_features(10))


   # print("\nTop 10 most informative words:")
   # for item in classifier.most_informative_features()[:10]:
   #      print(item[0])
   #
        # Sample input reviews
   input_reviews = [
       "It is worst movie",
       "This is a dull movie. I would never recommend it to anyone.",
       "The cinematography is pretty great in this movie",
       "The direction was terrible and the story was all over the place",
       "I hated the film. It was a disaster. Poor direction, bad acting.",
       "It was a wonderful and amazing movie. I loved it. Best direction, good acting.",
       "This movie dissapointed utterly"
   ]

   print("\nPredictions:")
   count =1
   for review in input_reviews:
       print("\nReview:", count, review)
       probdist = classifier.prob_classify(extract_features(review.split()))
       pred_sentiment = probdist.max()
       print(probdist)
       print("Predicted sentiment:", pred_sentiment)
       print("Probability:", round(probdist.prob(pred_sentiment), 2))
       count = count+1



