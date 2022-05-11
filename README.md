# Predicting Income Off of General Census Data, a STAT 159 Final Project
## Authors: Kavin Suresh, Wen-Ching (Naomi) Tu, George McIntire, Winston Cai

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/UCB-stat-159-s22/hw07-group24/HEAD)

### Description:
This project is themed on Census data. The Census data was obtained from UCI Machine Learning Repository. The underlying idea behind the project is to predict income off of various provided Census data, such as age, type of employer, years spent in education, etc. We created 4 different approaches to solve this problem (Version 1, Version 2, Version 3, and Version 4). For each approach we performed Exploratory Data Analysis to figure out the trends and ideas behind our dataset, did feature engineering to optimize features for our models, ran our models, and then tested their performance. The purpose of this project is to figure out the best practices to solving this income prediction problem and come up with a final prediction that optimizes by combining the predictions across the four different approaches. We aim to see if the combined approach yields better results.

The UCI Machine Learning Repository link to the data can be found here: http://archive.ics.uci.edu/ml/datasets/Census+Income.

### Quick Information of How to Run:
* **Environment Setup:** In order to set up the environment quickly and easily, simply open Terminal, make sure you are in this project's directory, and run the command "make env". This will quickly get the environment for the notebook up and running.
* **Running Analysis:** In order to run the analysis, the easiest way would be to use the Makefile! Simply open Terminal, make sure you are in this project's directory, and run the command "make all". This will run all the notebooks en masse. You can also manually go through the notebooks and run the cells through there for a more granular approach.
* **Using our Python package, projecttools:** Install the projecttools module by executing the `pip install .` command from the project home directory. In order to use the functions located within our local Python package, projecttools, simply run the following "from projecttools.utils import [function name]" and then the function can be used whenever, wherever!
* **Testing projecttools:** In order to run tests on project tools, simply run the test_utils.py function by running the following command once you have terminal open and are in the directory: `pytest projecttools`.
