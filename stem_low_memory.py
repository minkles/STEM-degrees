# The goal of this project is to build a logistic regression model to predict 
# the likelihood of working in a STEM (Science, Technology, Engineering, and Math) 
# career, based on basic demographics: Age, sex, race, state of origin.

print("Importing packages")

import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import numpy as np
import sqlite3
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from pandas.io import sql
from matplotlib.ticker import ScalarFormatter

print("Loading the raw data")

# Select only the relevant columns
pop_cols = ['AGEP','SEX','HISP','POBP','RAC1P','SCIENGP','SOCP']

# Load in all of the ACS2013 data
df = pd.concat([pd.read_csv("../input/pums/ss13pusa.csv", usecols=pop_cols),
  pd.read_csv("../input/pums/ss13pusb.csv", usecols=pop_cols)])

print("Cleaning data")

# The categorical variables in the data are all coded as integers. 
# For ease of interpretation, I will recode each of them with category labels.

# Recode place of birth variable with state names
oldNewMap = {1: "Alabama", 2: "Alaska", 4: "Arizona", 5: "Arkansas", 6: "California",
8: "Colorado", 9: "Connecticut", 10: "Delaware", 11: "District_of_Columbia",
12: "Florida", 13: "Georgia", 15: "Hawaii", 16: "Idaho", 17: "Illinois", 18: "Indiana",
19: "Iowa", 20: "Kansas", 21: "Kentucky", 22: "Louisiana", 23: "Maine", 24: "Maryland",
25: "Massachusetts", 26: "Michigan", 27: "Minnesota", 28: "Mississippi", 29: "Missouri",
30: "Montana", 31: "Nebraska", 32: "Nevada", 33: "New_Hampshire", 34: "New_Mexico",
35: "New_Jersey", 36: "New_York", 37: "North_Carolina", 38: "North_Dakota", 39: "Ohio",
40: "Oklahoma", 41: "Oregon", 42: "Pennsylvania", 44: "Rhode_Island", 45: "South_Carolina",
46: "South_Dakota", 47: "Tennessee", 48: "Texas", 49: "Utah", 50: "Vermont", 51: "Virginia",
53: "Washington", 54: "West_Virginia", 55: "Wisconsin", 56: "Wyoming"}
df['State'] = df['POBP'].map(oldNewMap)

# recode hispanic origin into 7 categories:
# Mexican, Puerto Rican, Cuban, Spaniard, Other Central American, South American, All Other Hispanic
# Read my blog post for how I arrived at these categories in my exploratory analysis:
# https://michaelinkles.wordpress.com/2016/04/03/modeling-inequality-in-stem-2/
oldNewMap = {1: "Not Spanish/Hispanic/Latino", 2: "Mexican", 3: "Puerto_Rican", 4: "Cuban", 
            5: "Other_Central_American", 6: "Other_Central_American", 7: "Other_Central_merican", 
            8: "Other_Central_American", 9: "Other_Central_American", 10: "Other_Central_American", 
            11: "Other_Central_American", 12: "Other_Central_American", 13: "South_American",
            14: "South_American", 15: "South_American", 16: "South_American", 17: "South_American", 
            18: "South_American", 19: "South_American", 20: "South_American", 21: "South_American", 
            22: "South_American", 23: "Spaniard", 24: "All_Other_Hispanic"}       
df['hisp_recode'] = df['HISP'].map(oldNewMap)

# recode sex variable
# First, change the integers that are used for coding.
# Right now, male = 1 and female = 2.
# to fit in a regression model they should be recoded to 0 and 1
if np.min(df['SEX']) > 0: #ensures that code won't be run if it's been recoded already
  df['SEX'] = df['SEX'] - 1
# Next, create a new column with Male/Female labels
oldNewMap = {0: "Male", 1: "Female"}
df['sex_recode'] = df['SEX'].map(oldNewMap)

# Recode race
# I will be using different categories than are used in the census data.
# All individuals of Hispanic origin will be categorized as Hispanic.
# Non-Hispanic White, Non-Hispanic Black and Asian will be included as categories.
# All other races are coded as "Other"
def race_recode(row):
  if row['HISP'] > 1:
    return "Hispanic"
  elif row['RAC1P'] == 1:
    return "White"
  elif row['RAC1P'] == 2:
    return "Black"
  elif row['RAC1P'] == 6:
    return "Asian"
  else:
    return "Other"
df['race_recode'] = df.apply(race_recode, axis=1)

# Recode outcome variables
# There are two outcomes variables i'm interested in using:
# "Has STEM degree" and "Works in STEM career"
# Each one must be represented as a binary

# Create "has science degree" outcome variable
oldNewMap = {1: 1, 2: 0}
df['science_degree'] = df['SCIENGP'].map(oldNewMap)
df['science_degree'].fillna(value=0,inplace=True)

# Create "works in STEM career" outcome variable
# STEM career categorization based on SOC codes from list found here:
# http://www.bls.gov/soc/Attachment_C_STEM.pdf

science_job_codes = ['113021','119041','119121','151111','151121','151122','151131','151132','151133',
                          '151134','151141','151142','151143','151151','151152','151199','152011','152021',
                          '152031','152041','152099','171021','171022','172011','172021','172031','172041',
                          '172051','172061','172071','172072','172081','172111','172112','172121','172131',
                          '172141','172151','172161','172171','172199','173012','173013','173019','173021',
                          '173022','173023','173024','173025','173026','173027','173029','173031','191011',
                          '191012','191012','191021','191022','191023','191029','191031','191032','191041',
                          '191042','191099','192011','192012','192021','192031','192032','192041','192042',
                          '192043','192099','194011','194021','194031','194041','194051','194091','194092',
                          '194093','251021','251022','251032','251041','251042','251043','251051','251052',
                          '251053','251054','414011','419031']
df['science_occupation'] = df['SOCP'].isin(science_job_codes).astype(int)

# add intercept for use in logistic regression model
df['intercept'] = 1.0

# In my exploratory analysis, I found there to be a strong interaction between
# sex and age, particularly in the "has science degree" outcome.
# For men, the likelihood of having a STEM degree did not vary much by age.
# Younger women, however, were far more likely to have STEM degrees than older women.
# Thus, the the gender gap (in having STEM degrees)
# is much larger at older ages and nonexistant under age 35.
# This is addressed in the following post in the "Results by Sex" section:
# https://michaelinkles.wordpress.com/2016/04/03/modeling-inequality-in-stem-2/
df['sex_age'] = df['SEX'] * df['AGEP']

print("Creating dummy variables")

# Pandas has a great function get_dummies that does all the work of
# creating dummy variables, and then you just have to join them.
# HOWEVER, this is memory intensive, so I will use a less memory intensive
# method of creating small tables for the dummy variables,
# and merging them using SQL

# state dummy variables
# Pennsylvania was omitted as the reference category due to its being
# a fairly large state that scored close to the average on outcome
states_to_include = ['Alabama','Alaska','Arizona','Arkansas','California',
                     'Colorado','Connecticut','Delaware','District_of_Columbia','Florida',
                     'Georgia','Hawaii','Idaho','Illinois','Indiana','Iowa',
                     'Kansas','Kentucky','Louisiana','Maine','Maryland','Massachusetts',
                     'Michigan','Minnesota','Mississippi','Missouri','Montana',
                     'Nebraska','Nevada','New_Hampshire','New_Jersey','New_Mexico',
                     'New_York','North_Carolina','North_Dakota','Ohio','Oklahoma',
                     'Oregon','Rhode_Island','South_Carolina','South_Dakota',
                     'Tennessee','Texas','Utah','Vermont','Virginia','Washington',
                     'West_Virginia','Wisconsin','Wyoming']
# create blank data frame (and add back Pennsylvania)
dummy_states = pd.DataFrame(index=(states_to_include+['Pennsylvania']),columns=states_to_include)
# for each state, when the index equals the column, include a 1
for state in states_to_include:
  dummy_states.set_value(state, state, 1)
# fill in zeroes for all other variables
dummy_states.fillna(value=0,inplace=True)

# Now, we do the same with race, using White as reference category

races_to_include = ['Black','Hispanic','Asian','Other']
dummy_race = pd.DataFrame(index=(races_to_include+['White']),columns=races_to_include)
for race in races_to_include:
  dummy_race.set_value(race, race, 1)
dummy_race.fillna(value=0,inplace=True)

# For Hispanic origin, Non-Hispanic is the reference category.
# In my final models, these will replace 'Hispanic' because they are more predictive.
# For some insight as to why different groups of Hispanics should not be treated
# as a single race in this model, I've covered that in another blog post:
# https://michaelinkles.wordpress.com/2016/03/12/whats-special-about-florida/
hisp_to_include = ['Mexican','Puerto_Rican','Cuban','Other_Central_American',
'South_American','Spaniard','All_Other_Hispanic']
dummy_hisp = pd.DataFrame(index=(hisp_to_include+['Not Spanish/Hispanic/Latino']),columns=hisp_to_include)
for hisp in hisp_to_include:
  dummy_hisp.set_value(hisp, hisp, 1)
dummy_hisp.fillna(value=0,inplace=True)

print("Joining dummy variables to data")

# To join the dummy variables, I created a SQLite database to store each table,
# then combined them into one big data frame using a SQL query

# Connect to SQLite
conn = sqlite3.connect('dat-test.db')

df.to_sql('df_main',con=conn,if_exists='replace',index=False)

dummy_states.to_sql('states',con=conn,if_exists='replace',index=True, index_label='State')

dummy_race.to_sql('races',con=conn,if_exists='replace',index=True, index_label='Race')

dummy_hisp.to_sql('hispanic_origins',con=conn,if_exists='replace',index=True, index_label='Hisp')

# In creating the final dataframe I'm using for the model,
# I want to subset out only people born in the 50 states or DC (so I can use State of Birth)
# who are over age 22 (so they are old enough to have a college degree)
# I could have done this earlier, but the most efficient way is to make it
# part of the SQL query that I'm doing anyway.
model_df = sql.read_sql(
"""
SELECT a.intercept, a.AGEP, a.SEX, a.sex_age, r.Asian, r.Black, r.Other, h.Mexican,
h.Puerto_Rican, h.Cuban, h.Spaniard, h.South_American, h.Other_Central_American, 
h.All_Other_Hispanic, s.Alabama, s.Alaska, s.Arizona, s.Arkansas, s.California,
s.Colorado, s.Connecticut,s.Delaware,s.District_of_Columbia,s.Florida,
s.Georgia,s.Hawaii,s.Idaho,s.Illinois,s.Indiana,s.Iowa,
s.Kansas,s.Kentucky,s.Louisiana,s.Maine,s.Maryland,s.Massachusetts,
s.Michigan,s.Minnesota,s.Mississippi,s.Missouri,s.Montana,
s.Nebraska,s.Nevada,s.New_Hampshire,s.New_Jersey,s.New_Mexico,
s.New_York,s.North_Carolina,s.North_Dakota,s.Ohio,s.Oklahoma,
s.Oregon,s.Rhode_Island,s.South_Carolina,s.South_Dakota,
s.Tennessee,s.Texas,s.Utah,s.Vermont,s.Virginia,s.Washington,
s.West_Virginia,s.Wisconsin,s.Wyoming, a.science_degree, a.science_occupation, a.POBP
FROM df_main as a
JOIN states as s
ON a.State = s.State
JOIN races as r
ON a.race_recode = r.Race
JOIN hispanic_origins as h
on a.hisp_recode = h.Hisp
WHERE a.AGEP > 22 and a.POBP < 60
""", con=conn)

print("Creating models")

# I'm creating two models here.
# The first model is the likelihood of having a STEM degree based on demographics.
# The second model is the likelihood of a person who has a STEM degree also
# having a job in a STEM career, based on demographics.
# This way, I will be able to analyze how much of the underrepresentaton of
# certain groups in STEM careers is based on underrepresentation in education,
# compared to post-education factors.

# For STEM degree model, include Age, Sex, Age*Sex interaction,
# race including detailed hispanic origin, and state of birth
train_cols_degree = ['intercept','AGEP','SEX','sex_age','Asian','Black','Other',
'Mexican','Puerto_Rican','Cuban','Spaniard','South_American',
'Other_Central_American','All_Other_Hispanic'] + states_to_include

# Fit final degree model
lm_final_degree = LogisticRegression()
lm_final_degree.fit(model_df[train_cols_degree],model_df['science_degree'])

# For STEM occupation model, I removed the states because they 
# turned out to not be significantly predictive.
# My hypothesis is this:
# State of birth predicts how likely someone is to get a college degree in STEM.
# Once they have that degree, their state of birth has little effect on how 
# likely they are to get a job in STEM.
train_cols_occ = ['intercept','AGEP','SEX','sex_age','Asian','Black','Other',
'Mexican','Puerto_Rican','Cuban','Spaniard','South_American',
'Other_Central_American','All_Other_Hispanic']
# Also, this model will need to be built on a subset of the original sample,
# which only contains peoople who have degrees in STEM
df_degree = model_df[model_df['science_degree']==1]

# Fit final occupation model
lm_final_occ = LogisticRegression()
lm_final_occ.fit(df_degree[train_cols_occ],df_degree['science_occupation'])

print("Generating ROC Curves")

# ROC curve for final degree model
probas_degree = lm_final_degree.predict_proba(model_df[train_cols_degree])
fig, ax = plt.subplots()
plt.title('ROC Curve for Science Degree Model')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.plot(roc_curve(model_df[['science_degree']], probas_degree[:,1])[0], 
  roc_curve(model_df[['science_degree']], probas_degree[:,1])[1])
plt.savefig('roc_degree.png')
plt.close()

# ROC Curve for final occupation model
probas_occ = lm_final_occ.predict_proba(df_degree[train_cols_occ])
fig, ax = plt.subplots()
plt.title('ROC Curve for Science Occupation Model')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.plot(roc_curve(df_degree[['science_occupation']], probas_occ[:,1])[0], 
  roc_curve(df_degree[['science_occupation']], probas_occ[:,1])[1])
plt.savefig('roc_occupation.png')
plt.close()

print("Plotting odds ratios for race")

# Right now, the models are fit in sklearn, which doesn't provide confidence intervals
# I want CIs in my plot, so I will have to use statsmodels to refit the models

# fit degree model
logit_degree = sm.Logit(model_df['science_degree'], model_df[train_cols_degree]) 

# create dataframe of CIs
result_degree = logit_degree.fit()
params_degree = result_degree.params
conf_degree = np.exp(result_degree.conf_int())
conf_degree['OR'] = np.exp(params_degree)
conf_degree.columns = ['2.5%', '97.5%', 'OR']

# add error column to degree CI dataframe, for use in plotting error bars
conf_degree['error'] = conf_degree['97.5%'] - conf_degree['OR']

# subset only the variables for race
race_odds_ratios = conf_degree[4:14]
                            
# add a new row for reference category
race_odds_ratios.loc['White'] = [1,1,1,0]

# sort by odds ratio, lowest to highest
race_odds_ratios = race_odds_ratios.sort_values(by='OR', ascending=True)

# Graph odds ratios for science degree
ind = np.arange(len(race_odds_ratios)) # how many bars
width = 0.7 # width of bars
fig, ax = plt.subplots()
ax.barh(ind, race_odds_ratios['OR'], width, color='lightblue', xerr=race_odds_ratios['error'])
plt.title('Odds Ratios for Science Degree')
plt.yticks(ind + width/2., race_odds_ratios.index.tolist()) # add category labels
plt.xscale('log') # plot on log scale
ax.get_xaxis().set_major_formatter(ScalarFormatter()) # convert x axis labels to scalar format
plt.xticks([0.5,1,2,3]) # add ticks at these values
plt.savefig('odds_ratios_race_degree.png')
plt.close()

# Repeat the whole thing for the occupation model

# fit occupation model
logit_occ = sm.Logit(df_degree['science_occupation'], df_degree[train_cols_occ]) 

# create dataframe of CIs
result_occ = logit_occ.fit()
params_occ = result_occ.params
conf_occ = np.exp(result_occ.conf_int())
conf_occ['OR'] = np.exp(params_occ)
conf_occ.columns = ['2.5%', '97.5%', 'OR']

# add error column to ocupation CI dataframe, for use in plotting error bars
conf_occ['error'] = conf_occ['97.5%'] - conf_occ['OR']
race_odds_ratios_occ = conf_occ[4:14]

# add a new row for reference category
race_odds_ratios.loc['White'] = [1,1,1,0]
race_odds_ratios_occ = race_odds_ratios_occ.sort_values(by='OR', ascending=True)                               

# Graph odds ratios for science occupation model
ind = np.arange(len(race_odds_ratios_occ)) # how many bars
width = 0.7 # width of bars
fig, ax = plt.subplots()
ax.barh(ind, race_odds_ratios_occ['OR'], width, color='lightblue', xerr=race_odds_ratios_occ['error'])
plt.title('Odds Ratios for Getting a STEM Job with a STEM Degree')
plt.yticks(ind + width/2., race_odds_ratios.index.tolist()) # add category labels
plt.xscale('log') # plot on log scale
ax.get_xaxis().set_major_formatter(ScalarFormatter()) # convert x axis labels to scalar format
plt.xticks([0.5,1,2,3]) # add ticks at these values
plt.savefig('odds_ratios_race_occupation.png')
plt.close()

print("Plotting odds ratios for state")

# The same process can be used for plotting odds ratios of the states.

# subset only the variables for state
state_odds_ratios = conf_degree[14:64]
                            
# add a new row for reference category
state_odds_ratios.loc['Pennsylvania'] = [1,1,1,0]

# sort by odds ratio, lowest to highest
state_odds_ratios = state_odds_ratios.sort_values(by='OR', ascending=True)

# Graph odds ratios for science degree
ind = np.arange(len(state_odds_ratios)) # how many bars
width = 0.5 # width of bars
fig, ax = plt.subplots()
ax.barh(ind, state_odds_ratios['OR'], width, color='lightblue', xerr=state_odds_ratios['error'])
plt.title('Odds Ratios for Science Degree')
plt.yticks(ind + width/2., state_odds_ratios.index.tolist(), fontsize= 6) # add category labels
plt.xscale('log') # plot on log scale
ax.get_xaxis().set_major_formatter(ScalarFormatter()) # convert x axis labels to scalar format
plt.xticks([0.5,1,2,3]) # add ticks at these values
plt.savefig('odds_ratios_state_degree.png')
plt.close()

print("Plotting predicted probabilities for age and sex")

print("Generating new dataframe for predictions")

# For age and sex, rather than plotting odds ratios, I'll plot predicted results,
# since they interact. Odds ratios are not meaningful on their own.

# Create a new dataframe for predictions
# For this dataframe, race and state will be held constant
# (at White and Utah, which yields close to average results),
# and age and sex will vary
predict_df = pd.DataFrame(index=range(0,146),columns=train_cols_degree)
predict_df[['intercept']] = 1
predict_df[['Utah']] = 1

# fill dataframe with all combinations
ages = range(23,96) 
i = 0
for age in ages:
  for sex in [0,1]:
    predict_df['AGEP'][i] = age
    predict_df['SEX'][i] = sex
    i += 1

# recreate interaction variable
predict_df['sex_age'] = predict_df['AGEP']*predict_df['SEX']

# fill all NA values with 0
predict_df.fillna(value=0,inplace=True)

print("Predicting outcomes")

# predict outcome for each combination
predict_df['degree_predict'] = lm_final_degree.predict_proba(predict_df[train_cols_degree])[:,1]
predict_df['occ_predict'] = lm_final_occ.predict_proba(predict_df[train_cols_occ])[:,1]

print("Plotting probabilities")

# STEM Degrees
sex_age_chart_degree = predict_df.pivot_table(index='AGEP',values='degree_predict',columns='SEX',aggfunc="mean")
sex_age_chart_degree.columns = ['Male','Female']
sex_age_chart_degree.index.name = 'Age'
sex_age_chart_degree.plot(title='Predicted Probability for STEM Degree')
plt.savefig('sex_age_degree.png')
plt.close()
plt.ylim(0,0.18)

# STEM Occupations
sex_age_chart_occ = predict_df.pivot_table(index='AGEP',values='occ_predict',columns='SEX',aggfunc="mean")
sex_age_chart_occ.columns = ['Male','Female']
sex_age_chart_occ.index.name = 'Age'
sex_age_chart_occ.plot(title='Predicted Probability for STEM Occupation with STEM Degree')
plt.savefig('sex_age_occupation.png')
plt.close()
plt.ylim(0,0.18)