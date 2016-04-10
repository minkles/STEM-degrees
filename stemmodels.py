# Load required libraries and functions
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from bokeh.io import output_file, show, gridplot
from bokeh.plotting import figure, show
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.basemap import Basemap as Basemap
from matplotlib.colors import rgb2hex
from matplotlib.patches import Polygon
from matplotlib.colors import LinearSegmentedColormap


# Load the saved, formatted CSV
df = pd.read_csv("df_formatted.csv")
df.reset_index()
df = df.drop('Unnamed: 0', 1)

# Exploratory analysis

# Create graph of science degree rates by age and sex
q = df.pivot_table(index='AGEP',values='science_degree',columns="sex_recode",
	aggfunc="mean")
q.columns.name = "sex"
q.index.name = 'Age'
q.plot(title='Percent of Population with Science Degrees')
plt.savefig('images/exploratory_sex_age_degree.png')
plt.close()

# Create graph of science degree rates by age and sex (college graduates only)
z = df[df['college_degree']==1].pivot_table(index='AGEP',values='science_degree',
	columns="sex_recode",aggfunc="mean")
z.index.name = 'Age'
z.columns.name = "sex"
z.plot(title='Percent of College Graduates with STEM degrees')
plt.savefig('images/exploratory_sex_age_degree_2.png')
plt.close()

# Create graph of STEM career rates by age and sex
l = df.pivot_table(index='AGEP',values='science_occupation',columns="sex_recode",
	aggfunc="mean")
l.columns.name = "sex"
l.index.name = 'Age'
l.plot(title='Percent of Population with Science Careers')
plt.savefig('images/exploratory_sex_age_career.png')
plt.close()

# Create graphs of STEM career rates by age and race, for each state 
# To avoid small sample sizes, a race will only be included in a state's graph
# if there are more than 1000 people of that race in the state included in this
# sample. Only states that include at least one substantial minority population 
# will be included in the interactive graph.

# create list of states to include in exploratory graphs
race_counts = df.pivot_table(values='science_degree',index='State',columns=['race_recode'],aggfunc="count")
include_hispanic = race_counts[race_counts['Hispanic']>1000]
include_black = race_counts[race_counts['Black']>1000]
include_asian = race_counts[race_counts['Asian']>1000]
states_in_model = race_counts[(race_counts['Black']>1000) | (race_counts['Asian']>1000) | (race_counts['Hispanic']>1000)]
data = df.pivot_table(values='science_degree',index=['State','age_recode'],columns=['race_recode'],aggfunc="mean")

# create a function to make the graphs
def make_plot(state):
	'''
	Takes state name as input, and returns a plot of age and science degrees for that state,
	stratified by race.
	Only races with 1000 or more observations in the sample for that state will be included.
	'''
	state_name = str(state)
	p = figure(plot_width=300, plot_height=300, title=state_name, tools='pan', title_text_font_size="14pt")
	current_state = data.loc[state_name]
	current_state['age_group'] = [0,1,2,3,4,5,6,7,8,9]
	p.title = state_name
	if states_in_model.loc[state_name]['Asian'] > 1000:
	    p.line(current_state['age_group'], current_state['Asian'], color="orange", legend='Asian')
	if states_in_model.loc[state_name]['White'] > 1000:
	    p.line(current_state['age_group'], current_state['White'], color="blue", legend='White')
	if states_in_model.loc[state_name]['Hispanic'] > 1000:
	    p.line(current_state['age_group'], current_state['Hispanic'], color="green", legend='Hispanic')
	if states_in_model.loc[state_name]['Black'] > 1000:
	    p.line(current_state['age_group'], current_state['Black'], color="red", legend='Black')
	p.xaxis.axis_label = 'Age Group'
	p.yaxis.axis_label = 'Percent with Science Degrees'
	p.xaxis.axis_label_text_font_size = "10pt"
	p.yaxis.axis_label_text_font_size = "10pt"
	p.legend.label_text_font_size = "7pt"
	return p

# Make the graphs
p_al = make_plot('Alabama'); p_az = make_plot('Arizona') ; p_ar = make_plot('Arkansas')
p_ca = make_plot('California') ; p_co = make_plot('Colorado') ; p_ct = make_plot('Connecticut')
p_dc = make_plot('District of Columbia') ; p_fl = make_plot('Florida') ; p_ga = make_plot('Georgia')
p_hi = make_plot('Hawaii') ; p_il = make_plot('Illinois') ; p_in = make_plot('Indiana')
p_ky = make_plot('Kentucky') ; p_la = make_plot('Louisiana') ; p_md = make_plot('Maryland')
p_ma = make_plot('Massachusetts') ; p_mi = make_plot('Michigan') ; p_ms = make_plot('Mississippi')
p_mo = make_plot('Missouri') ; p_nj = make_plot('New Jersey') ; p_nm = make_plot('New Mexico')
p_ny = make_plot('New York') ; p_nc = make_plot('North Carolina') ; p_oh = make_plot('Ohio')
p_ok = make_plot('Oklahoma') ; p_pa = make_plot('Pennsylvania') ; p_sc = make_plot('South Carolina')
p_tn = make_plot('Tennessee') ; p_tx = make_plot('Texas') ; p_va = make_plot('Virginia')

output_file = "images/exploratory_stateplots.png"
plot = gridplot([[p_al,p_az,p_ar,p_ca,p_co,p_ct],[p_dc,p_fl,p_ga,p_hi,p_il,p_in],
                 [p_ky,p_la,p_md,p_ma,p_mi,p_ms],[p_mo,p_nj,p_nm,p_ny,p_nc,p_oh],
                 [p_ok,p_pa,p_sc,p_tn,p_tx,p_va]], toolbar_location=None)
show(plot)




# Creating the model 

# add intercept 
df['intercept'] = 1.0

# create dummy variables for state
dummy_states = pd.get_dummies(df['State'])
df = df.join(dummy_states)

# create dummy variables for race
dummy_race = pd.get_dummies(df['race_recode'])
df = df.join(dummy_race)

# create dummy variables for age group
dummy_age = pd.get_dummies(df['age_recode'])
dummy_age.columns = ['23-28','29-34','35-40','41-46','47-52',
                     '53-58','59-64','65-70','71-76','77+']
df = df.join(dummy_age)

# create dummy variables for hispanic origin
dummy_hisp = pd.get_dummies(df['hisp_recode'])
df = df.join(dummy_hisp)

# Create interaction variable for sex and age
def sex_age(row):
	if row['SEX'] == 0:
		return 0
	elif row['SEX'] == 1:
		return row['AGEP']

interaction = df.apply(sex_age, axis=1)

df['sex_age'] = interaction

# create models
states_to_include = ['Alabama','Alaska','Arizona','Arkansas','California',
                     'Colorado','Connecticut','Delaware','District of Columbia','Florida',
                     'Georgia','Hawaii','Idaho','Illinois','Indiana','Iowa',
                     'Kansas','Kentucky','Louisiana','Maine','Maryland','Massachusetts',
                     'Michigan','Minnesota','Mississippi','Missouri','Montana',
                     'Nebraska','Nevada','New Hampshire','New Jersey','New Mexico',
                     'New York','North Carolina','North Dakota','Ohio','Oklahoma',
                     'Oregon','Rhode Island','South Carolina','South Dakota',
                     'Tennessee','Texas','Utah','Vermont','Virginia','Washington',
                     'West Virginia','Wisconsin','Wyoming'] #PA excluded as reference

train_cols_degree = ['intercept','AGEP','SEX','sex_age','Asian','Black','Other',
'Mexican','Puerto Rican','Cuban','Spaniard','South American',
'Other Central American','All other Spanish/Hispanic/Latino'] + states_to_include

train_cols_occ = ['intercept','AGEP','SEX','sex_age','Asian','Black','Other',
'Mexican','Puerto Rican','Cuban','Spaniard','South American',
'Other Central American','All other Spanish/Hispanic/Latino']

# Fit final degree model
lm_final_degree = LogisticRegression()
lm_final_degree.fit(df[train_cols_degree],df['science_degree'])

# Fit final occupation model
# include only the subset of the original sample that has science degrees
df_degree = df[df['science_degree']==1]
lm_final_occ = LogisticRegression()
lm_final_occ.fit(df_degree[train_cols_occ],df_degree['science_occupation'])

# ROC curve for final degree model
probas_degree = lm_final_degree.predict_proba(df[train_cols_degree])
plt.plot(roc_curve(df[['science_degree']], probas_degree[:,1])[0], 
	roc_curve(df[['science_degree']], probas_degree[:,1])[1])
plt.savefig('images/roc_degree.png')
plt.close()

# ROC Curve for final occupation model
probas_occ = lm_final_occ.predict_proba(df[train_cols_occ])
plt.plot(roc_curve(df[['science_occupation']], probas_occ[:,1])[0], 
	roc_curve(df[['science_occupation']], probas_occ[:,1])[1])
plt.savefig('images/roc_occupation.png')
plt.close()

# fit all in statsmodels for confidence intervals

# fit degree model
logit_degree = sm.Logit(df['science_degree'], df[train_cols_degree]) 

# create dataframe of CIs
result_degree = logit_degree.fit()
params_degree = result_degree.params
conf_degree = np.exp(result_degree.conf_int())
conf_degree['OR'] = np.exp(params_degree)
conf_degree.columns = ['2.5%', '97.5%', 'OR']

# add error column to degree CI dataframe, for use in plotting error bars
conf_degree['error'] = conf_degree['97.5%'] - conf_degree['OR']
race_odds_ratios = conf_degree[4:14]
                            
# add a new row for reference category
race_odds_ratios.loc['White'] = [1,1,1,0]
race_odds_ratios = race_odds_ratios.sort_values(by='OR', ascending=True)

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
plt.savefig('images/odds_ratios_race_degree.png')
plt.close()

# Graph odds ratios for science degree
ind = np.arange(len(race_odds_ratios_occ)) # how many bars
width = 0.7 # width of bars
fig, ax = plt.subplots()
ax.barh(ind, race_odds_ratios_occ['OR'], width, color='lightblue', xerr=race_odds_ratios_occ['error'])
plt.title('Odds Ratios for Getting a STEM Job with a STEM Degree')
plt.yticks(ind + width/2., race_odds_ratios.index.tolist()) # add category labels
plt.xscale('log') # plot on log scale
ax.get_xaxis().set_major_formatter(ScalarFormatter()) # convert x axis labels to scalar format
plt.xticks([0.5,1,2,3]) # add ticks at these values
plt.savefig('images/odds_ratios_race_occupation.png')
plt.close()

# Analysis of sex and age resuls
# Because these two variables have an interaction, 
# I will be analyzing predicted probabilities instead of odds ratios.

# Create a new dataframe for predictions
race_cols = train_cols_degree[4:14] + ['White']
state_cols = train_cols_degree[14:64] + ['Pennsylvania']
ages = range(25,96,5) #instead of using every possible age I'll go up in increments of 5
predict_df = pd.DataFrame(index=range(0,16830),columns=np.array(['intercept','AGEP','SEX','sex_age','race_recode','State']))
predict_df[['AGEP','SEX','sex_age']] = 0
predict_df[['intercept']] = 1
predict_df[['race_recode']] = 'none'
predict_df[['State']] = 'none'

# fill dataframe with all combinations
i=0
for state in state_cols:
	for race in race_cols:
		for age in ages:
			for sex in [0,1]:
				predict_df['AGEP'][i] = age
				predict_df['SEX'][i] = sex
				predict_df['sex_age'][i] = sex*age
				predict_df['race_recode'][i] = race
				predict_df['State'][i] = state
				i += 1

# recreate dummy variables
#create dummy variables for state
#dummy_states_2 = pd.get_dummies(predict_df['State'])
#predict_df = predict_df.join(dummy_states_2)

# create dummy variables for race
#dummy_race_2 = pd.get_dummies(predict_df['race_recode'])
#predict_df = predict_df.join(dummy_race_2)

# predict outcome for each combination
#predict_df['degree_predict'] = lm_final_degree.predict_proba(predict_df[train_cols_degree])[:,1]
#predict_df['occ_predict'] = lm_final_occ.predict_proba(predict_df[train_cols_occ])[:,1]

# This chart shows predicted probability for a white person from Utah 
# having a science degree, by sex and age.
# Utah/White combination was chosen for being as close as possible to average
#df_sex_age_chart = predict_df[(predict_df['race_recode']=='White')&(predict_df['State']=='Utah')]
#sex_age_chart = df_sex_age_chart.pivot_table(index='AGEP',
#	values='degree_predict',columns='SEX',aggfunc="mean")
#sex_age_chart.columns = ['Male','Female']
#sex_age_chart.index.name = 'Age'
#sex_age_chart.plot(title='Predicted Probability for Science Degree')
#plt.ylim(0,0.18)
#plt.savefig('images/results_sex_degree.png')
#plt.close()

# This chart shows predicted probability for a white person from Utah 
# having a science occupation, given that they have a science degree
#df_sex_age_chart = predict_df[(predict_df['race_recode']=='White')&(predict_df['State']=='Utah')]
#sex_age_chart = df_sex_age_chart.pivot_table(index='AGEP',values='occ_predict',columns='SEX',aggfunc="mean")
#sex_age_chart.columns = ['Male','Female']
#sex_age_chart.index.name = 'Age'
#sex_age_chart.plot(title='Predicted Probability for STEM Occupation with STEM Degree')
#plt.savefig('images/results_sex_occupation.png')
#plt.close()

# Map of odds ratios by state
# Subset out state odds ratios and add back reference category
states = conf_degree[14:64]
states.loc['Pennsylvania'] = [1,1,1,0]
states = states.sort_values(by='OR')

# Most of the following code comes directly from a matplotlib example.
# Source: https://github.com/matplotlib/basemap/blob/master/examples/fillstates.py

plt.figure(figsize=(20,20))

# Lambert Conformal map of lower 48 states.
m = Basemap(llcrnrlon=-119,llcrnrlat=22,urcrnrlon=-64,urcrnrlat=49,
            projection='lcc',lat_1=33,lat_2=45,lon_0=-95)
# draw state boundaries.
# data from U.S Census Bureau
# http://www.census.gov/geo/www/cob/st2000.html
shp_info = m.readshapefile('shapefiles/st99_d00','states',drawbounds=True)
# choose a color for each state based on population density.
colors={}
statenames=[]
# create custom colormap
cdict1 = {'red':   ((0.0, 1.0, 1.0),
                    (0.5, 0.85, 0.85),
                    (1.0, 0.0, 0.0)),

         'green': ((0.0, .25, 0.25),
                   (0.5, 0.85, 0.85),
                   (1.0, 0.65, 0.65)),

         'blue':  ((0.0, 0.25, 0.25),
                   (0.5, 0.1, 0.1),
                   (1.0, 0.0, 0.0))
         }

redyellowgreen = LinearSegmentedColormap('redyellowgreen', cdict1)
cmap = redyellowgreen
vmin = 0.5; vmax = 1.5 # set range.
for shapedict in m.states_info:
	statename = shapedict['NAME']
	# skip DC
	if statename not in ['District of Columbia','Puerto Rico']:
		odds_ratio = states['OR'][statename]
		# calling colormap with value between 0 and 1 returns
		# rgba value.  Invert color range (hot colors are high
		# population), take sqrt root to spread out colors more.
		colors[statename] = cmap((odds_ratio-vmin)/(vmax-vmin))[:3]
	statenames.append(statename)
# cycle through state names, color each one.
ax = plt.gca() # get current axes instance
for nshape,seg in enumerate(m.states):
	# skip DC and Puerto Rico.
	if statenames[nshape] not in ['District of Columbia','Puerto Rico']:
		color = rgb2hex(colors[statenames[nshape]]) 
		poly = Polygon(seg,facecolor=color,edgecolor=color)
		ax.add_patch(poly)
# draw meridians and parallels.
m.drawparallels(np.arange(25,65,20),labels=[1,0,0,0])
m.drawmeridians(np.arange(-120,-40,20),labels=[0,0,0,1])
plt.savefig('images/state_odds_ratios_map.png')
plt.close()
