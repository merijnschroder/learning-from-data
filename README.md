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
To run the program in this mode, use the `--generate-dataset` flag.
This flag needs at least to be accompanied by `--test-data` and `--model-path`.
Here, `--model-path` specifies the pickle file of the saved model.

When running in this mode, the program
1. extracts the most important predictor words for offensive tweets from the model specified in the `--model-path`,
2. loads the training, development, and testing datasets and randomly adds one of these words to all non-offensive tweets, and
3. writes the adapted data to `data/generated`.

### Print Dataset Statistics

