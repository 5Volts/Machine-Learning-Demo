![alt text](https://raw.githubusercontent.com/5Volts/Machine-Learning-Demo/master/img.jpg)

# Machine Learning Demo
This repository contain the Python Script for visualizing decision boundaries in classification and best fit line in regression.

# Get Started
## Dependencies
You'll need the following modules:
`sklearn`
`matplotlib`
`numpy`

If not, just type `pip install sklearn matplotlib numpy` into your commandline.

# Running the project
## Regression
You can simply run `python regression.py` it will randomly generate data suited for regression solution. The default algorithm is Decision Tree.

There are a couple of arguments you can parse.

`--algorithm`

There are multiple algorithm to choose from : 
`adaboost` `decisiontree` `randomforest` `kneighbors` `extratrees` `svr` `mlp` 
Default would be decisiontree. All but support vector regressor will remain in its default settings. Support Vector Regressor has been given a linear kernel.

Ex: `python regression.py --algorithm adaboost`

`--n_samples`

Number of randomly generated datapoint. Default is 20

`--noise`

Noise and inconsistency in dataset. Default is 20

`--dark_mode`

Set to true if you want a graph with a black background.

`--resolution`

Number of connecting points that will construct the best fit line. Default is 20.

## Classification
You can simply run `python classification.py` it will randomly generate data suited for classification solution. Each datapoint has 2 features and 1 label, either 0 or 1.

The default algorithm is Decision Tree.

There are a couple of arguments you can parse.

`--algorithm`

There are multiple algorithm to choose from : 
`adaboost` `decisiontree` `randomforest` `kneighbors` `extratrees` `svr` `mlp` 
Default would be decisiontree. All algorithms are at it's default settings.

Ex: `python regression.py --algorithm randomforest`

`--n_samples`

Number of randomly generated datapoint. Default is 100

`--dark_mode`

Set to true if you want a graph with a black background.

`--resolution`

Resolution of the colored background. A higher resolution results in a more detailed image of the background coloring but takes longer to generate. Default is 20

`--use_proba`

If true, instead of just two solid colors in the background, it returns a colored graph with varying intensity according to the probabilities. Default is false.