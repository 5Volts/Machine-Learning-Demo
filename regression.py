'''
This is a Python Scripts that visualize the logic behind a few type of Regressors.
You can choose from 7 different algorithms and it will be trained on a dataset with 1 feature
and 1 label in each datapoint.
'''
from sklearn.datasets.samples_generator import make_regression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from matplotlib import pyplot
import argparse
import numpy as np

def give_best_fit(X,reg, reso):
  '''
  This function will run the regressor against the minimum X value in the data until
  the maximum X value and create the coordinates to plot the best fit line.

  :param X: 1D Data
  Ex: [0,1,2,3,4]

  :param reg: Trained Scikit-Learn Regressor
  Ex: DecisionTreeRegressor()

  :param reso: Number of best-fit coordinate to plot
  Ex: 20

  :return: X and Y Coordinates for best fit
  Ex: ([0,1,2,3], [2,3,4,5])
  '''
  maximum = max(X)
  minimum = min(X)
  interval = (maximum - minimum) / reso
  x_best = []
  y_best = []
  for pred in range(reso + 1):
    single_data = minimum + pred * interval
    x_best.append(single_data)
    ans = reg.predict(single_data)[0]
    y_best.append(ans)
  return x_best,y_best

def graph(X,Y,xbest,ybest,reg,dark_mode):
  '''
  This function helps with generating the graph
  :param X: List of X datapoint
  Ex : [0,1,2,3,4]

  :param Y: List of Y labels
  Ex : [1,2,3,4,5]

  :param xbest: List of coordinates in X-axis for best fit
  Ex : [0,1,2,3,4]

  :param ybest: List of coordinates in Y-axis for best fit
  Ex : [1,2,3,4,5]

  :param reg: Scikit-Learn Regressor
  Ex : DecisionTreeRegressor()

  :param dark_mode: Toggle for dark background
  Ex : True

  :return: None
  '''
  if dark_mode:
    pyplot.style.use('dark_background')
  pyplot.title(str(reg).partition('(')[0])
  pyplot.scatter(X, Y, color='blue', s=30)
  pyplot.plot(xbest, ybest, 'r')
  pyplot.show()

def main(args):
  '''
  Main Function
  :param args: Dictionary from Parser
  :return: None
  '''
  dict_arg = vars(args)
  algo = dict_arg['algorithm']
  dark_mode = dict_arg['dark_mode']
  resolution = dict_arg['resolution']
  n_samples = dict_arg['n_samples']
  noise = dict_arg['noise']

  X, Y = make_regression(n_samples=n_samples, n_features=1, noise=noise)

  hash_function = {'decisiontree':DecisionTreeRegressor(),
                   'adaboost':AdaBoostRegressor(),
                   'randomforest': RandomForestRegressor(),
                   'kneighbors': KNeighborsRegressor(),
                   'extratrees': ExtraTreesRegressor(),
                   'svr': SVR(kernel='linear'),
                   'mlp': MLPRegressor()}

  reg = hash_function[algo]

  reg.fit(X, Y)

  X_reshaped = np.reshape(X, (len(X)))

  x_best, y_best = give_best_fit(X_reshaped, reg, reso=resolution)

  graph(X, Y, x_best, y_best, reg, dark_mode=dark_mode)


if __name__ == '__main__':
  parser = argparse.ArgumentParser('Regression Demo by 5Volts')

  parser.add_argument('--algorithm',type=str,
                      default='decisiontree',
                      help='There are multiple algorithm to choose from.'
                           'adaboost decisiontree randomforest kneighbors extratrees'
                           'svr mlp . Default would be decisiontree. All but support '
                           'vector regressor will remain in its default settings.'
                           'Support Vector Regressor has been given a linear kernel')

  parser.add_argument('--n_samples',type=int,
                      default=20,
                      help='Number of random generated data')

  parser.add_argument('--noise', type=int,
                      default=20,
                      help='Noise in dataset')

  parser.add_argument('--dark_mode',type=bool,
                      default=False,
                      help='For all you dark mode lovers out there. Set to True'
                           'if you want the matplotlib graph to have a black '
                           'background')

  parser.add_argument('--resolution',type=int,
                      default=20,
                      help='This is the resolution in which the best fit line will'
                           'be plotted. A higher value will result in more resolution'
                           'in the best fit line but takes longer to plot out')

  args = parser.parse_args()
  main(args)