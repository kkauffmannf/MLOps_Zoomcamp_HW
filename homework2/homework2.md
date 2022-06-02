## Homework 2

## Q1. 

What's the version that you have?

<b> mlflow, version 1.26.0 </b>

## Q2. 

How many files were saved to `OUTPUT_FOLDER`?

<b>4</b>


## Q3.

How many parameters are automatically logged by MLflow?

<b>17</b>


## Q4. 

In addition to `backend-store-uri`, what else do you need to pass to properly configure the server?

<b>`default-artifact-root`</b>


## Q5. 

What's the best validation RMSE that you got?


<b>6.628</b>


## Q6. Promote the best model to the model registry

The results from the hyperparameter optimization are quite good so we can assume that we are ready to test some of these models in production. In this exercise, you'll promote the best model to the model registry. We have prepared a script called `register_model.py`, which will check the results from the previous step and select the top 5 runs. After that, it will calculate the RMSE of those models on the test set (March 2021 data) and save the results to a new experiment called `random-forest-best-models`.

Your task is to update the script `register_model.py` so that it selects the model with the lowest RMSE on the test set and registers it to the model registry.

Tip 1: you can use the method `search_runs` from the `MlflowClient` to get the model with the lowest RMSE.
Tip 2: to register the model you can use the method `mlflow.register_model` and you will need to pass the right model_uri in the form of a string that looks like this: `"runs:/<RUN_ID>/model"`, and the name of the model (make sure to choose a good one!).

What is the test RMSE of the best model?

* 6.1
* 6.55
* 7.93
* 15.1


## Submit the results

Submit your results here: https://forms.gle/9wXF5ntBA3FNe65L9

It's possible that your answers won't match exactly. If it's the case, select the closest one.


## Deadline

The deadline for submitting is 31 May 2022 (Tuesday) at 17:00 CET. After that, the form will be closed.


## Solution

The solution will be put here after the deadline.
