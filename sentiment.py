__author__ = 'shekhargulati'

import os, re, math, collections
import nltk.metrics
from nltk.classify import NaiveBayesClassifier

POLARITY_DATA_DIR = os.path.join('polarity-data', 'rt-polaritydata')
POSITIVE_REVIEWS = os.path.join(POLARITY_DATA_DIR, 'rt-polarity-pos.txt')
NEGATIVE_REVIEWS = os.path.join(POLARITY_DATA_DIR, 'rt-polarity-neg.txt')


def evaluate_features(feature_select):
    pos_features = []
    neg_features = []

    for line in open(POSITIVE_REVIEWS, 'r'):
        pos_words = re.findall(r"[\w']+|[.,!?;]", line.rstrip())
        pos_features.append([feature_select(pos_words), 'pos'])

    for line in open(NEGATIVE_REVIEWS, 'r'):
        neg_words = re.findall(r"[\w']+|[.,!?;]", line.rstrip())
        neg_features.append([feature_select(neg_words), 'neg'])

    print("len of positive features %d" % len(pos_features))
    pos_cutoff = int(math.floor(len(pos_features) * 3 / 4))
    neg_cutoff = int(math.floor(len(neg_features) * 3 / 4))

    print("pos_cutoff %d neg_cutoff %d" % (pos_cutoff, neg_cutoff))

    training_data = pos_features[:pos_cutoff] + neg_features[:neg_cutoff]
    test_data = pos_features[pos_cutoff:] + neg_features[neg_cutoff:]

    classifier = NaiveBayesClassifier.train(training_data)

    reference_set = collections.defaultdict(set)
    test_set = collections.defaultdict(set)

    for index, (features, label) in enumerate(test_data):
        reference_set[label].add(index)
        predicted = classifier.classify(features)
        test_set[predicted].add(index)

    print 'train on %d instances, test on %d instances' % (len(training_data), len(test_data))
    print 'accuracy:', nltk.classify.util.accuracy(classifier, test_data)
    print 'pos precision:', nltk.metrics.precision(reference_set['pos'], test_set['pos'])
    print 'pos recall:', nltk.metrics.recall(reference_set['pos'], test_set['pos'])
    print 'neg precision:', nltk.metrics.precision(reference_set['neg'], test_set['neg'])
    print 'neg recall:', nltk.metrics.recall(reference_set['neg'], test_set['neg'])
    classifier.show_most_informative_features(10)


#creates a feature selection mechanism that uses all words
def make_full_dict(words):
    return dict([(word, True) for word in words])

#tries using all words as the feature selection mechanism
print 'using all words as features'
evaluate_features(make_full_dict)









