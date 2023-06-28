## DEPLOYING MODEL INFRRENCE AS A WEBSERVICE

There are various ways of deploying a Machine Learning Model one of those is Deploying as a Web Service.
While deploying this model as a web service I have followed the following steps
    * Creating a virtual environment -- Particularly for this deployment scheme I have used Pipenv
    * Creating a script for Predicting -- In this step I have written 3 functions
        * 1 --- "prepare_features" this function takes the a json feature load and prepares them in the same way I have encoded the pickup and dropoff location features during model development.
        * 2 --- "predict_features" this function takes features prepared by the function above and applies the dictionary vectorizer and the model which are shiped from the original notebook.
        * 3 --- "predict_endpoint" is the actual end point which exposes the model inference and returns trip duration in minutes.