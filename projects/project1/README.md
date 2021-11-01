# Machine Learning Project 1-Fall 2021

## Description

This project is part of the EPFL Machine Learning course for fall 2021 and was implemented by Younes Moussaif, Pauline Conti and Clémence Barsi

The aim of this project is to implement 6 machine learning models using solely the numpy library and to optimize accuracy on the test set using pre-processing steps. To obtain our test accuracy, we uploaded our submissions to AIcrowd.


## How to use

To run our best performing model, the script ```run.py``` must be ran having the files test.csv and train.csv in a folder Data. The predictions will be generated in an output.csv file.


## Content of our project 

- ```cross_validation.py``` contains all our methods used for cross-validation
- ```eda_preprocessing.py``` contains our methods used to explore the data and to pre-process the data, handling invalid values, redundant features...
- ```grid_search.py``` contains the functions we created to select the hyper-parameters that achieved the best accuracy
- ```implementations.py``` has the definition of the 6 models asked and the methods needed to run them
- ```pipeline.py``` has the pipeline used for our best performing model
- ```plots.py``` contains the code to create the plots included in the report
- ```proj1_helpers.py``` are the helpers provided in the template of the project to create our submission on AIcrowd
- ```run.py``` is the script that creates the best predictions in the test set.
