## Setting Up the Environment
Make sure to have Python 3.7 installed on your machine.
You can check this by running

```ps
> python -V
Python 3.7.x
```

It is easiest to set up a virtual environment to run the code. 
```ps
> python -m venv .venv
```
Make sure to activate the virtual environment.

To install the dependencies, run
```ps
> python -m pip install --upgrade pip
> python -m pip install -r requirements.txt
```

## Running the Program
This program can be ran in four different 'modes'.

### Train and Evaluate

### Grid-Search

### Generate Dataset
When running in this mode, the program
1. extracts the most important predictor words for offensive tweets from the model specified in the `--model-path`,
2. loads the training, development, and testing datasets specified by the arguments and randomly adds one of these words to all non-offensive tweets, and
3. writes the adapted data to `data/generated`.

Note that this only needs to happen once, and not every time you want to run a model on generated data.
When the data is generated by this mode, just change the data file locations to `data/generated`.

When running the program with `--generate-dataset`, you always need to add `--model-path` with the path of the pickle file of the model you want to use the most important features from.
Note that this model needs to have a `coef_` attribute, which generally is the case for linear models in scikit-learn.

Also the path of the testing dataset that will be adapted should be specified with `--test-data`.
`--train-data` and `--dev-data` have default values, but if you want to change those, you also need to specify them.

### Print Dataset Statistics

