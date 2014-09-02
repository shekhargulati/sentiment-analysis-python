__author__ = 'shekhargulati'

"""
1. Read the positive review file, tokenize it, and add it to a list of tuple. Tuple ([list of words],label)
2. Read the negative review file, tokenize it, and add it to a list of tuple. Tuple ([list of words],label)
3. Create training and test data from the actual data. Training data 75% and Test Date 25%
4. Train the classifer
5. Classify using the classifier
"""


def tokenize_file_and_apply_label(filename, label):
    words_label_tuple_list = []
    for line in open(filename, 'r').readlines():
        words = [word.lower() for word in line.split() if len(word) >= 3]
        words_label_tuple_list.append((list_to_dict(words), label))
    return words_label_tuple_list


def list_to_dict(words):
    return dict([(word, True) for word in words])


def get_training_data(pos_tokens, neg_tokens, cutoff):
    import math

    pos_cutoff = int(math.floor(cutoff * len(pos_tokens)))
    neg_cutoff = int(math.floor(cutoff * len(neg_tokens)))
    return pos_tokens[:pos_cutoff] + neg_tokens[:neg_cutoff]


def get_test_data(pos_tokens, neg_tokens, cutoff):
    import math

    pos_cutoff = int(math.floor(cutoff * len(pos_tokens)))
    neg_cutoff = int(math.floor(cutoff * len(neg_tokens)))
    return pos_tokens[pos_cutoff:] + neg_tokens[neg_cutoff:]


def all_words_in_training_data(training_data):
    all_words = []
    for item in training_data:
        for word in item[0]:
            all_words.append(word)
    return all_words


if __name__ == "__main__":
    import os

    BASE_DIR = os.path.join("polarity-data", "rt-polaritydata")
    POSITIVE_REVIEWS_FILE = os.path.join(BASE_DIR, "rt-polarity-pos.txt")
    pos_tokens = tokenize_file_and_apply_label(POSITIVE_REVIEWS_FILE, "positive")

    NEGATIVE_REVIEWS_FILE = os.path.join(BASE_DIR, "rt-polarity-neg.txt")
    neg_tokens = tokenize_file_and_apply_label(NEGATIVE_REVIEWS_FILE, "negative")

    training_data = get_training_data(pos_tokens, neg_tokens, 1)
    test_data = get_test_data(pos_tokens, neg_tokens, 0.75)

    print "Training Data %d, Test Data %d" % (len(training_data), len(test_data))

    # word_features = get_word_features(all_words_in_training_data(training_data))
    # feature_training_set = feature_training_set(word_features, training_data)

    from nltk.classify import NaiveBayesClassifier

    classifier = NaiveBayesClassifier.train(training_data)

    import collections

    expected_set = collections.defaultdict(set)
    actual_set = collections.defaultdict(set)

    for index, item in enumerate(test_data):
        expected_set[item[1]].add(index)
        sentiment = classifier.classify(item[0])
        actual_set[sentiment].add(index)

    import nltk

    print 'accuracy: %.2f' % nltk.classify.util.accuracy(classifier, test_data)
    print 'pos precision: %.2f' % nltk.metrics.precision(expected_set['positive'], actual_set['positive'])
    print 'pos recall: %.2f' % nltk.metrics.recall(expected_set['positive'], actual_set['positive'])
    print 'neg precision: %.2f' % nltk.metrics.precision(expected_set['negative'], actual_set['negative'])
    print 'neg recall: %.2f' % nltk.metrics.recall(expected_set['negative'], actual_set['negative'])












