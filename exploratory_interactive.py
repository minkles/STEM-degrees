from os.path import join, dirname
import numpy as np
import pandas as pd
from bokeh.models import ColumnDataSource, DataRange1d, Range1d, VBox, HBox, Select
from bokeh.palettes import Blues4
from bokeh.plotting import Figure, show
from bokeh.io import output_file, curdoc
from scipy.signal import savgol_filter

# Load the saved, formatted CSV
df = pd.read_csv("df_formatted.csv")
df.reset_index()
df = df.drop('Unnamed: 0', 1)

# These are all the data that will change when a new menu option is selected
STATISTICS = ['White','Black','Asian','Hispanic','Male','Female','Total']

# Function for reloading the data every time a new menu option is selected
def get_dataset(src, name, counts, selection):
    state_name = str(name)
    # Set up data by creating a pivot table for how the data is being grouped
    if state_name == "All States":
        if selection == 'Race':
            df2 = src.pivot_table(values='science_degree',index=['age_recode'],columns=['race_recode'],aggfunc="mean").reset_index()
        elif selection == 'Sex':
            df2 = src.pivot_table(values='science_degree',index=['age_recode'],columns=['sex_recode'],aggfunc="mean").reset_index()
        else:
            df2 = src.pivot_table(values='science_degree',index=['age_recode'],aggfunc="mean").reset_index()
    else:
        if selection == 'Race':
            df2 = src.pivot_table(values='science_degree',index=['State','age_recode'],columns=['race_recode'],aggfunc="mean").reset_index()
        elif selection == 'Sex':
            df2 = src.pivot_table(values='science_degree',index=['State','age_recode'],columns=['sex_recode'],aggfunc="mean").reset_index()
        else:
            df2 = src.pivot_table(values='science_degree',index=['State','age_recode'],aggfunc="mean").reset_index()
        # subset only the state being used
        df2 = df2[df2.State == state_name].copy()
    # Label each age group with the center
    df2['age_group'] = [25,31,37,43,49,55,61,67,73,80]
    # Make age group the index so it's automatically the x axis
    df2 = df2.set_index(['age_group'])
    df2.sort_index(inplace=True)
    if selection == 'Race':
        # set all unused columns to -1 so that they don't appear in the graph
        # however these columns will still need to be carried over to ColumnDataSource
        df2['Totl'] = -1
        df2['Male'] = -1
        df2['Female'] = -1
        df2.fillna(-1, inplace=True)
        if state_name != "All States":
            # Any race that has <1000 people in the sample for that given state will not be included
            # due to small sample size
            for race in ['Asian','Black','White','Hispanic']:
                if counts.loc[state_name][race] < 1000:
                    df2[race] = -1
    elif selection == 'Sex':
        df2['White'] = -1
        df2['Black'] = -1
        df2['Asian'] = -1
        df2['Hispanic'] = -1
        df2['Totl'] = -1
    elif selection == 'Totals Only':
        df2['White'] = -1
        df2['Black'] = -1
        df2['Asian'] = -1
        df2['Hispanic'] = -1
        df2['Male'] = -1
        df2['Female'] = -1
        df2['Totl'] = df2['science_degree']
    return ColumnDataSource(data=df2)

def make_plot(source, title):
    plot = Figure(plot_width=1000, plot_height=600, tools="", toolbar_location=None, y_range=[0,0.5])
    plot.title = title
    
    # Each possible line must be created
    # In the current version of Bokeh, there is no way to hide or show lines interactively, which is
    # why the values had to be set to -1 in the get_dataset function
    plot.line(x='age_group', y='Asian', color="cornflowerblue", 
        source=source, legend="Asian", line_width=2)
    plot.line(x='age_group', y='White', color="green", 
        source=source, legend="White", line_width=2)
    plot.line(x='age_group', y='Black', color="darkviolet", 
        source=source, legend="Black", line_width=2)
    plot.line(x='age_group', y='Hispanic', color="darkgoldenrod", 
        source=source, legend="Hispanic", line_width=2)
    plot.line(x='age_group', y='Male', color="blue", 
        source=source, legend="Male", line_width=2)
    plot.line(x='age_group', y='Female', color="crimson", 
        source=source, legend="Female", line_width=2)
    plot.line(x='age_group', y='Totl', color="grey", 
        source=source, legend="Total", line_width=2)
    
    # fixed attributes
    plot.xaxis.axis_label = 'Age'
    plot.yaxis.axis_label = 'Percent with Science Degrees'
    plot.xaxis.axis_label_text_font_size = "10pt"
    plot.yaxis.axis_label_text_font_size = "10pt"
    plot.legend.label_text_font_size = "7pt"

    return plot

# set up callbacks
def update_plot(attrname, old, new):
    # When anything is changed, this function updates the title of the plot, 
    # re-runs get_dataset, and replaced the old data with new data
    state = state_select.value
    selection = data_select.value
    plot.title = states[state]['name']
    src = get_dataset(df, states[state]['name'], counts, selection)
    for key in STATISTICS:
        source.data.update(src.data)
        
# set up initial data
state = 'ALL'
selection = "Race"

states = {
    'ALL': {
        'name': "All States",
    },
    'Alabama': {
        'name': 'Alabama',
    },
    'Alaska': {
        'name': 'Alaska',
    },
    'Arizona': {
        'name': 'Arizona',
    },
    'Arkansas': {
        'name': 'Arkansas'
    },
    'California': {
        'name': 'California',
    },
    'Colorado': {
        'name': 'Colorado',
    },
    'Connecticut': {
        'name': 'Connecticut',
    },
    'Delaware': {
        'name': 'Delaware',
    },
    'District of Columbia': {
        'name': 'District of Columbia',
    },
    'Florida': {
        'name': 'Florida',
    },
    'Georgia': {
        'name': 'Georgia',
    },
    'Hawaii': {
        'name': 'Hawaii',
    },
    'Idaho': {
        'name': 'Idaho',
    },
    'Illinois': {
        'name': 'Illinois',
    },
    'Indiana': {
        'name': 'Indiana',
    },
    'Iowa': {
        'name': 'Iowa',
    },
    'Kansas': {
        'name': 'Kansas',
    },
    'Kentucky': {
        'name': 'Kentucky',
    },
    'Louisiana': {
        'name': 'Louisiana',
    },
    'Maine': {
        'name': 'Maine',
    },
    'Maryland': {
        'name': 'Maryland',
    },
    'Massachusetts': {
        'name': 'Massachusetts',
    },
    'Michigan': {
        'name': 'Michigan',
    },
    'Minnesota': {
        'name': 'Minnesota',
    },
    'Mississippi': {
        'name': 'Mississippi',
    },
    'Missouri': {
        'name': 'Missouri',
    },
    'Montana': {
        'name': 'Montana',
    },
    'Nebraska': {
        'name': 'Nebraska',
    },
    'Nevada': {
        'name': 'Nevada',
    },
    'New Hampshire': {
        'name': 'New Hampshire',
    },
    'New Jersey': {
        'name': 'New Jersey',
    },
    'New Mexico': {
        'name': 'New Mexico',
    },
    'New York': {
        'name': 'New York',
    },
    'North Carolina': {
        'name': 'North Carolina',
    },
    'North Dakota': {
        'name': 'North Dakota',
    },
    'Ohio': {
        'name': 'Ohio',
    },
    'Oklahoma': {
        'name': 'Oklahoma',
    },
    'Oregon': {
        'name': 'Oregon',
    },
    'Pennsylvania': {
        'name': 'Pennsylvania',
    },
    'Rhode Island': {
        'name': 'Rhode Island',
    },
    'South Carolina': {
        'name': 'South Carolina',
    },
    'South Dakota': {
        'name': 'South Dakota',
    },
    'Tennessee': {
        'name': 'Tennessee',
    },
    'Texas': {
        'name': 'Texas',
    },
    'Utah': {
        'name': 'Utah',
    },
    'Vermont': {
        'name': 'Vermont',
    },
    'Virginia': {
        'name': 'Virginia',
    },
    'Washington': {
        'name': 'Washington',
    },
    'West Virginia': {
        'name': 'West Virginia',
    },
    'Wisconsin': {
        'name': 'Wisconsin',
    },
    'Wyoming': {
        'name': 'Wyoming',
    }
}

# create dropdown menus
state_select = Select(value=state, title='State', options=sorted(states.keys()))
data_select = Select(value=selection, title='Group By:', options=['Race','Sex','Totals Only'])

# counts dataframe used to determine if the sample size for a given race in a given state is large enough
counts = df.pivot_table(values='science_degree',index='State',columns=['race_recode'],aggfunc="count")

# get initial data
source = get_dataset(df, states[state]['name'], counts, selection)

# make the plot
plot = make_plot(source, states[state]['name'])

# add interactivity to menus
state_select.on_change('value', update_plot)
data_select.on_change('value', update_plot)

# set up layout of plot
controls = HBox(state_select, data_select)
curdoc().add_root(VBox(controls, plot))