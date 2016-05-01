# Modeling racial and gender inequality in STEM
Final Project for Data Science at General Assembly

This purpose of this project is to model the the likelihood of receiving a degree or finding employment in a STEM (science, technology, engineering, and math) field using logistic regression, and derive interesting conclusions from these models.

For a non-technical overview of the project and its results, read my most recent blog post on the project: [https://michaelinkles.wordpress.com/2016/04/03/modeling-inequality-in-stem-2/]("https://michaelinkles.wordpress.com/2016/04/03/modeling-inequality-in-stem-2/")

In order to run any code from this repo, the 2013 American Community Survey data must be downloaded from [https://www.kaggle.com/census/2013-american-community-survey]("https://www.kaggle.com/census/2013-american-community-survey") to the STEM-degrees folder.

This repo contains the following:

* __stem_low_memory.py__: The most recent and most efficient version of the code, but lacks some of the detail of the research process that's in the iPython notebook 

* __finalproject-inkles.ipynb__: an iPython notebook containing all of my code along descriptions of the code in markdown, documenting my entire process

* __stemdata.py__: a python script for all of the data wrangling that needs to be done on the original data for the models to work. it outputs a file called df_formatted.csv which is used by stemmodels.py

* __stemmodels.py__: a python script that uses the output of stemdata.py to generate the models and output all of the graphs that are used in my blog post

* __images folder__: contains the output of stemmodels.py; all the images that are used in the blog post

*  __shapefiles folder__: shapefiles to be used by baseplot in generating the map of odds ratios

* __seri.csv__: an additional data file containing each state's SERI ([Science and Engineering Readiness Index]("https://www.aps.org/units/fed/newsletters/summer2011/white-cottle.cfm")) score, to be used with 2013 ACS data in the data wrangling step.

* __exploratory_interactive.py__ (new): Code for an interactive visualization of my exploratory analysis that displays a line graph of science degree rate by age. Results can be sorted by state, and grouped by race or sex. This code makes use of [Bokeh]([http://bokeh.pydata.org/en/latest/]) plotting library, which can be used to create D3.js style graphics in Python. I'm working on getting this running as a standalone web app, but for now the visualization can be viewed by taking the following steps:
	1. Run stemdata.py
	2. [Install bokeh]([http://bokeh.pydata.org/en/latest/docs/installation.html]) by typing "conda install bokeh" or "pip install bokeh"
	3. Type into the command line:  
		bokeh serve --show exploratory_interactive.py