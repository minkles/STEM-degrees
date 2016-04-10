import pandas as pd
import numpy as np

# Load in all of the ACS2013 data
raw_1 = pd.read_csv("pums/ss13pusa.csv")
raw_2 = pd.read_csv("pums/ss13pusb.csv")

# Load outside data - SERI scores
seri = pd.read_csv("SERI.csv")

# Select only the relevant columns and combine two halves of dataset
pop_cols = ['SERIALNO','SPORDER','PWGTP','AGEP','CIT','CITWP','SEX','ANC','ANC1P','ANC2P','DECADE',
            'FOD1P','FOD2P','HISP','NATIVITY','POBP','QTRBIR','RAC1P','RAC2P','RAC3P','SCHL',
            'SCIENGP','SCIENGRLP','WAOB','SOCP']
pop_1 = raw_1[pop_cols]
pop_2 = raw_2[pop_cols]
df = pd.concat([pop_1,pop_2])

# Merge state names and SERI scores by State into the main data frame
df = df.merge(seri, on='POBP',how='left')

# Subset only native born (50 states and DC) and aged 23 or older
df = df[df['AGEP']>22] #Age is 23 or older
df = df[df['POBP']<60] #Born in 50 states or DC

# The outcome variable ('SCIENGP') needs to be converted to a binary
def science_degree_binary(row):
	if row['SCIENGP'] == 1:
		return 1
	else:
		return 0
science_degree = df.apply(science_degree_binary, axis=1)
df['science_degree'] = science_degree

# Another outcome all college degrees - convert to binary
def college_degree_binary(row):
	if row['SCHL'] >= 21:
		return 1
	else:
		return 0
college_degree = df.apply(college_degree_binary, axis=1)
df['college_degree'] = college_degree

# Another outcome: science occupation
# A list of STEM Occupation SOC codes is found here:
# http://www.bls.gov/soc/Attachment_C_STEM.pdf
def science_occupation_binary(row):
	if row['SOCP'] in ['113021','119041','119121','151111','151121','151122','151131','151132','151133',
                          '151134','151141','151142','151143','151151','151152','151199','152011','152021',
                          '152031','152041','152099','171021','171022','172011','172021','172031','172041',
                          '172051','172061','172071','172072','172081','172111','172112','172121','172131',
                          '172141','172151','172161','172171','172199','173012','173013','173019','173021',
                          '173022','173023','173024','173025','173026','173027','173029','173031','191011',
                          '191012','191012','191021','191022','191023','191029','191031','191032','191041',
                          '191042','191099','192011','192012','192021','192031','192032','192041','192042',
                          '192043','192099','194011','194021','194031','194041','194051','194091','194092',
                          '194093','251021','251022','251032','251041','251042','251043','251051','251052',
                          '251053','251054','414011','419031']:
		return 1
	else:
		return 0    
science_occupation = df.apply(science_occupation_binary, axis=1)
df['science_occupation'] = science_occupation

# Recoding race to 5 categories
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
recoded_race = df.apply(race_recode, axis=1)
df['race_recode'] = recoded_race

# recode the HISP variable for easy readability
oldNewMap = {1: "Not Spanish/Hispanic/Latino", 2: "Mexican", 3: "Puerto Rican", 4: "Cuban", 
             5: "Dominican", 6: "Costa Rican", 7: "Guatemalan", 8: "Honduran", 9: "Nicaraguan",
            10: "Panamanian", 11: "Salvadorian", 12: "Other Central American", 13: "Argentinian",
            14: "Bolivian", 15: "Chilean", 16: "Colombian", 17: "Ecuadorian", 18: "Paraguayan",
            19: "Peruvian", 20: "Uruguayan", 21: "Venezuelan", 22: "Other South American",
            23: "Spaniard", 24: "All Other Spanish/Hispanic/Latino"}
df['hispanic_origin'] = df['HISP'].map(oldNewMap)

# Recode sex so male = 0 and female = 1
if np.min(df['SEX']) > 0: #ensures that code won't be run if it's been recoded already
	df['SEX'] = df['SEX'] - 1

# Create a new variable with labels for sex, to be used in exploratory analysis
oldNewMap = {0: "Male", 1: "Female"}
df['sex_recode'] = df['SEX'].map(oldNewMap)

# Create age ranges - 6 years in each range, plus another range for 77+
df['age_recode'] = pd.cut(df['AGEP'],bins=[20,28.5,34.5,40.5,46.5,52.5,58.5,64.5,70.5,76.6,100],
                          labels=['23-28','29-34','35-40','41-46','47-52','53-58','59-64','65-70','71-76','77+'])

# Recode detailed hispanic origin into smaller categories

def hispanic_recode(row):
	if row['HISP'] == 1:
		return "Not Spanish/Hispanic/Latino"
	elif row['HISP'] == 2:
		return "Mexican"
	elif row['HISP'] == 3:
		return "Puerto Rican"
	elif row['HISP'] == 4:
		return "Cuban"
	elif row['HISP'] <= 12:
		return "Other Central American"
	elif row['HISP'] <= 22:
		return 'South American'
	elif row['HISP'] == 23:
		return 'Spaniard'
	elif row['HISP'] == 24:
		return 'All other Spanish/Hispanic/Latino'
        
recoded_hisp = df.apply(hispanic_recode, axis=1)

df['hisp_recode'] = recoded_hisp

# Save the file so it can easily be loaded next time
df.to_csv("df_formatted.csv")