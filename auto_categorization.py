import os
import string
import random
import pickle
from nltk import word_tokenize
from nltk import FreqDist
from nltk.corpus import stopwords
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

BASE_DIR = '/Users/andoni/projects/train-py-bbc/bbc'
LABELS = ['business', 'entertainment', 'politics', 'sport', 'tech']
USELESS_WORDS = ['said', 'mr', 'also', 'us', 'year', 'years', 'would']



def get_splits(dataset):
  random.shuffle(dataset) # randomize dataset
  X_train = [] # training documents
  y_train = [] # corresponding training labels

  X_test = [] # test documents
  y_test = [] # corresponding test label

  dataset_length = len(dataset)
  pivot = int(.80 * dataset_length)
  for i in range(0, pivot):
    X_train.append(dataset[i][1])
    y_train.append(dataset[i][0])

  for i in range(pivot, dataset_length):
    X_test.append(dataset[i][1])
    y_test.append(dataset[i][0])
  return X_train, X_test, y_train, y_test


def evaluate_classifier(title, classifier, vectorizer, X_test, y_test):
  X_test_tfidf = vectorizer.transform(X_test)
  y_prediction = classifier.predict(X_test_tfidf)

  precision = metrics.precision_score(y_test, y_prediction, average='micro')
  recall = metrics.recall_score(y_test, y_prediction, average='micro')
  f1 = metrics.f1_score(y_test, y_prediction, average='micro')

  print('%s\t%f\t%f\t%f' % (title, precision, recall, f1))

def train_classifier(dataset):
  X_train, X_test, y_train, y_test = get_splits(dataset)

  # transform clean dataset into vectors
  vectorizer = CountVectorizer(analyzer='word', stop_words='english', ngram_range=(1, 3), min_df=3)

  # create matrix
  dataset_matrix = vectorizer.fit_transform(X_train)

  # create classifier
  naives_bayes_classifier = MultinomialNB().fit(dataset_matrix, y_train)

  # evaluate classifier
  evaluate_classifier('Naive Bayes \tTRAIN\t', naives_bayes_classifier, vectorizer, X_train, y_train)
  evaluate_classifier('Naive Bayes \tTEST\t', naives_bayes_classifier, vectorizer, X_test, y_test)

  # save the classifier
  classifier_filename = 'naive_bayes_clasifier.pkl'
  pickle.dump(naives_bayes_classifier, open(classifier_filename, 'wb'))

  # save the vectorizer to transform new data
  vectorizer_filename = 'count_vectorizer.pkl'
  pickle.dump(vectorizer, open(vectorizer_filename, 'wb'))

def get_stopwords():
  stop_words = set(stopwords.words('english'))
  for useless_word in USELESS_WORDS:
    stop_words.add(useless_word)
  return stop_words

def clean_dataset_item(item):
  # Remove punctuation
  text = item.translate(str.maketrans('', '', string.punctuation))
  # Convert to lower case
  text = text.lower()
  return text

def get_tokens(text):
  stop_words = get_stopwords()
  tokens = word_tokenize(text)
  tokens = [token for token in tokens if not token in stop_words]
  return tokens

def setup_dataset(file):
  dataset = []
  with open(file, 'r') as datafile:
    for row in datafile:
      columns = row.split('\t')
      label = columns[0]
      content = columns[2].replace('\n', '').strip()
      data_item = (label, content)
      dataset.append(data_item)
  return dataset


def get_frequency_distribution(dataset):
  tokens = defaultdict(list);
  for data_item in dataset:
    label = data_item[0]
    text = clean_dataset_item(data_item[1])
    words = get_tokens(text)
    tokens[label].extend(words)
  for label, tokens in tokens.items():
    print(label)
    frequency = FreqDist(tokens)
    print(frequency.most_common(20))


def classify(text, classifier_filename, vectorizer_filename):
  # load classifier
  classifier = pickle.load(open(classifier_filename, 'rb'))

  # vectorize the text
  vectorizer = pickle.load(open(vectorizer_filename, 'rb'))

  # predict with naives bayes classifier the vectorized text
  prediction = classifier.predict(vectorizer.transform([text]))
  print('#################')
  print('The following text: %s' % text)
  print('It was classified on %s' % prediction[0])
  print('#################')

def create_dataset(labels, source, dataset_filename):
  with open(dataset_filename, 'w') as outfile:
    outfile.write('%s\t%s\t%s\n' % ('LABEL', 'FILENAME', 'CONTENT'))
    for label in labels:
      dir = '%s/%s' % (source, label)
      for filename in os.listdir(dir):
        dir_filename = '%s/%s' % (dir, filename)
        with open(dir_filename, 'rb') as file:
          text = file.read().decode(errors = 'replace').replace('\n', '')
          outfile.write('%s\t%s\t%s\n' % (label, filename, text))

if __name__ == '__main__':
  # create_dataset(LABELS, BASE_DIR, 'test_first_dataset.txt')
  tuple_dataset = setup_dataset('test_first_dataset.txt')
  # get_frequency_distribution(tuple_dataset)
  train_classifier(tuple_dataset)

  # text_tech = "Ubisoft blames ‘technical error’ for showing pop-up ads in Assassin’s Creed. A fullscreen pop-up ad appeared in Assassin’s Creed Odyssey for some players this week."

  # text_politics = "Tens of thousands of people have marched through central London at a demonstration against antisemitism. Organisers estimated 60,000 took part in the first march of its kind since the Israel-Gaza war began, including former Prime Minister Boris Johnson."

  # text = "Tiger Woods signs for ‘rusty’ 75 on comeback at Hero World Challenge"

  # classify(text_tech, 'naive_bayes_clasifier.pkl', 'count_vectorizer.pkl')
  # classify(text_politics, 'naive_bayes_clasifier.pkl', 'count_vectorizer.pkl')
  # classify(text, 'naive_bayes_clasifier.pkl', 'count_vectorizer.pkl')

  print('Done')
