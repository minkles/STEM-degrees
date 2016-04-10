# Modeling racial and gender inequality in STEM
Final Project for Data Science at General Assembly

This purpose of this project is to model the the likelihood of receiving a degree or finding employment in a STEM (science, technology, engineering, and math) field using logistic regression, and derive interesting conclusions from these models.

For a non-technical overview of the project and its results, read my most recent blog post on the project: [https://michaelinkles.wordpress.com/2016/04/03/modeling-inequality-in-stem-2/]("https://michaelinkles.wordpress.com/2016/04/03/modeling-inequality-in-stem-2/")

In order to run any code from this repo, the 2013 American Community Survey data must be downloaded from [https://www.kaggle.com/census/2013-american-community-survey]("https://www.kaggle.com/census/2013-american-community-survey") to the STEM-degrees folder.

This repo contains the following:

* __finalproject-inkles.ipynb__: an iPython notebook containing all of my code along descriptions of the code in markdown, documenting my entire process

* __stemdata.py__: a python script for all of the data wrangling that needs to be done on the original data for the models to work. it outputs a file called df_formatted.csv which is used by stemmodels.py

* __stemmodels.py__: a python script that uses the output of stemdata.py to generate the models and output all of the graphs that are used in my blog post

* __images folder__: contains the output of stemmodels.py; all the images that are used in the blog post

*  __shapefiles folder__: shapefiles to be used by baseplot in generating the map of odds ratios

* __seri.csv__: an additional data file containing each state's SERI ([Science and Engineering Readiness Index]("https://www.aps.org/units/fed/newsletters/summer2011/white-cottle.cfm")) score, to be used with 2013 ACS data in the data wrangling step.