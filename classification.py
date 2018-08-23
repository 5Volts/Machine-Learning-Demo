'''
This is a Python Scripts that visualize the logic behind a few type of Classifiers.
You can choose from 7 different algorithms and it will be trained on a random dataset with 2 feature
and 1 label in each datapoint.
It allows for easier understanding, especially for algorithms like Decision Trees.
'''
from sklearn.datasets.samples_generator import make_classification
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from matplotlib import pyplot,cm
from pandas import DataFrame
import numpy as np
import argparse

def color_background_with_label(clf,res,use_proba):
  '''
  This function generates the two 2D arrays needed for coloring
  the graph background
  :param clf: Trained Scikit-Learn Classifier
  Ex: DecisionTreeClassifier()

  :param res: Resolution in which the graph is colored
  Ex : 20

  :param use_proba: Whether or not to use probabilities or actual labels.
  Using probabilities will result in a nice looking graph with varying
  color intensity generated according to the probabilities.

  :return: Two 2D arrays for coloring
  Ex : ([[0,1,1],
        [0,1,1],
        [0,1,1]]
        ,
        [[1,0,0],
        [1,0,0],
        [1,0,0]])
  '''
  scale = (-3, 3)
  d = scale[1] - scale[0]
  Y0 = np.zeros((res, res))
  Y1 = np.zeros((res, res))
  for ix in range(res):
    for iy in range(res):
      if use_proba:
        probs = clf.predict_proba([[scale[0] + ix * d / res, scale[1] - iy * d / res]])[0]
        label = np.argmax(probs)
        proba = np.max(probs)
      else:
        label = clf.predict([[scale[0] + ix * d / res, scale[1] - iy * d / res]])[0]
      if label == 0:
        Y0[iy][ix] = proba
        Y1[iy][ix] = None
      elif label == 1:
        Y1[iy][ix] = proba
        Y0[iy][ix] = None
  return Y0,Y1

def graph(X,L1Colored,L2Colored,clf,dark_mode,mid,use_proba):
  '''
  This function handles generating the graph
  :param X: Data with 2 features in each datapoint
  Ex : [[0,0],[0,1],[0,2],[0,3]]

  :param L1Colored: 2D array
  Ex : [[0,1,1],
        [0,1,1],
        [0,1,1]]

  :param L2Colored: 2D array
  Ex : [[1,0,0],
        [1,0,0],
        [1,0,0]]

  :param clf: Scikit-Learn Classifier
  Ex : DecisionTreeClassifier()

  :param dark_mode: Toggle for dark background
  Ex : True

  :param mid: Mid point in data that separates the two different classes.
  The data are sorted in a way that makes it easy to isolate into their
  respective classes.
  Ex : 50

  :param use_proba: Color with probabilities instead.
  Ex : True

  :return: None
  '''
  if dark_mode:
    pyplot.style.use('dark_background')
  pyplot.title(str(clf).partition('(')[0])
  pyplot.scatter(X[:mid, 0], X[:mid, 1], color='cyan', s=30)
  pyplot.scatter(X[mid:, 0], X[mid:, 1], color='magenta', s=30)
  if use_proba:
    pyplot.imshow(L1Colored, cmap=cm.plasma, interpolation='nearest', extent=(-3, 3, -3, 3))
    pyplot.imshow(L2Colored, cmap=cm.viridis, interpolation='nearest', extent=(-3, 3, -3, 3))
  else:
    pyplot.imshow(L1Colored, cmap=cm.Blues, interpolation='nearest', extent=(-3, 3, -3, 3))
    pyplot.imshow(L2Colored, cmap=cm.Purples, interpolation='nearest', extent=(-3, 3, -3, 3))
  pyplot.show()

def main(args):
  '''
  Main Function
  :param args: Dictionary from argparse
  :return: None
  '''
  dict_arg = vars(args)
  algo = dict_arg['algorithm']
  dark_mode = dict_arg['dark_mode']
  resolution = dict_arg['resolution']
  n_samples = dict_arg['n_samples']
  use_proba = dict_arg['use_proba']

  X, Y = make_classification(n_samples=n_samples, n_features=2, n_redundant=0)

  hash_function = {'decisiontree': DecisionTreeClassifier(),
                   'adaboost': AdaBoostClassifier(),
                   'randomforest': RandomForestClassifier(),
                   'kneighbors': KNeighborsClassifier(),
                   'extratrees': ExtraTreesClassifier(),
                   'svc': SVC(),
                   'mlp': MLPClassifier()}

  clf = hash_function[algo]

  indicies = np.argsort(Y)
  Y = Y[indicies]
  mid = np.argmax(Y)
  X = X[indicies]
  clf.fit(X, Y)

  c1, c2 = color_background_with_label(clf, resolution, use_proba)
  graph(X,c1,c2,clf,dark_mode,mid, use_proba)
  
if __name__ == '__main__':
  parser = argparse.ArgumentParser('Regression Demo by 5Volts')

  parser.add_argument('--algorithm', type=str,
                      default='decisiontree',
                      help='There are multiple algorithm to choose from.'
                           'adaboost decisiontree randomforest kneighbors extratrees'
                           'svr mlp . Default would be decisiontree.')

  parser.add_argument('--n_samples', type=int,
                      default=100,
                      help='Number of random generated data')

  parser.add_argument('--dark_mode', type=bool,
                      default=False,
                      help='For all you dark mode lovers out there. Set to True'
                           'if you want the matplotlib graph to have a black '
                           'background')

  parser.add_argument('--resolution', type=int,
                      default=20,
                      help='This is the resolution in which the colored background will'
                           'be drawn. A higher value will result in more details'
                           'in the image but takes longer to create.')

  parser.add_argument('--use_proba',type=bool,
                      default=False,
                      help='Set to true if you want the graph to be colored according to probabilities'
                           'instead of labels.')

  args = parser.parse_args()
  main(args)




