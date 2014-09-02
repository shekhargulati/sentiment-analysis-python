__author__ = 'shekhargulati'

pos_tweets = [('I love this car', 'positive'),
              ('This view is amazing', 'positive'),
              ('I feel great this morning', 'positive'),
              ('I am so excited about the concert', 'positive'),
              ('He is my best friend', 'positive'),
              ('This movie was great', 'positive'),
              ('This movie was not pathetic', 'positive')]

neg_tweets = [('I do not like this car', 'negative'),
              ('This view is horrible', 'negative'),
              ('I feel tired this morning', 'negative'),
              ('I am not looking forward to the concert', 'negative'),
              ('He is my enemy', 'negative'),
              ('This is a pathetic movie', 'negative')]

tweets_with_sentiment = []
for (tweet, sentiment) in pos_tweets + neg_tweets:
    filtered_tweet_words = [word.lower() for word in tweet.split() if len(word) >= 3]
    tweets_with_sentiment.append((filtered_tweet_words, sentiment))

print tweets_with_sentiment

all_words = []
for words, sentiment in tweets_with_sentiment:
    all_words.extend(words)

import nltk

fd = nltk.FreqDist(all_words)
word_features = fd.keys()

print word_features

# Extract Features


def extract_features(document):
    unique_words_in_document = set(document)
    features = {}
    for word_feature in word_features:
        features['contains(%s)' % word_feature] = (word_feature in unique_words_in_document)
    return features


training_set = nltk.classify.apply_features(extract_features, tweets_with_sentiment)

print training_set

classifier = nltk.classify.NaiveBayesClassifier.train(training_set)

classifier.show_most_informative_features(10)

test_tweet = "#RajaNatwarlal is a pathetic movie"

features_test_tweet = extract_features(test_tweet.split())
print features_test_tweet
print classifier.classify(features_test_tweet)

