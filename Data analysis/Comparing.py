import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import plotly.graph_objs as go


import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('Downloads\MSDhoni.csv')
df1= pd.read_csv('Downloads\msdvsteams.csv')
df2= pd.read_csv('Downloads\msdinground.csv')
df6= pd.read_csv('Downloads\msposition.csv')
df8= pd.read_csv('Downloads\MSWINKNOCKS.csv')
df10= pd.read_csv('Downloads\msresult.csv')

df3 = pd.read_csv('Downloads\dkvsopp.csv')
df4 = pd.read_csv('Downloads\data\dk.csv')
df5 = pd.read_csv('Downloads\data\dkinground.csv')
df7= pd.read_csv('Downloads\dkposition.csv')
df9= pd.read_csv('Downloads\msposition.csv')


#1.msdmatchesvs teams
sns.barplot(x='Matches', y='Teams', data=df1)
plt.xlabel('Matches')
plt.ylabel('Teams')
plt.title('Grouped bar plot of Matches vs Teams')
plt.show()

#2.msdinningssvs teams
df = df.dropna()
sns.barplot(x='Innings', y='Teams', data=df1)
plt.xlabel('Innings')
plt.ylabel('Teams')
plt.title('Grouped bar plot of Innings vs Teams')
plt.show()

#3.msdrunssvs teams
df = df.dropna()
sns.barplot(x='Runs', y='Teams', data=df1)
plt.xlabel('Runs')
plt.ylabel('Teams')
plt.title('Grouped bar plot of Runs vs Teams')
plt.show()

#4.msdstrikeratevs teams
Df_info=df.copy()

match_df = pd.concat([df['Strike Rate']]).value_counts().reset_index()
match_df.set_axis(['Strike Rate','Teams'],axis = 'columns' ,inplace = True)

def get_color(name, number):
    pal = list(sns.color_palette(palette=name, n_colors=number).as_hex())
    return pal

pal_vi = get_color('viridis_r', len(match_df))

fig = px.bar(df, x='Strike Rate', y='Teams', text='Strike Rate',
             color ='Strike Rate', color_discrete_sequence=pal_vi)
fig.update_layout(
    title={
        'text': "Strike Rate per Teams",
        'y':1,
        'x':0.45,
        'xanchor': 'center',
        'yanchor': 'top'})
fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig.update_layout({'plot_bgcolor': 'white',
                   'paper_bgcolor': 'white'})
fig.update_layout(width=1000, height=600,
             
#5.msdaveragevs teams
Df_info=df.copy()

match_df = pd.concat([df['Average']]).value_counts().reset_index()
match_df.set_axis(['Average','Teams'],axis = 'columns' ,inplace = True)

def get_color(name, number):
    pal = list(sns.color_palette(palette=name, n_colors=number).as_hex())
    return pal

pal_vi = get_color('viridis_r', len(match_df))

fig = px.bar(df, x='Average', y='Teams', text='Average',
             color ='Average', color_discrete_sequence=pal_vi)
fig.update_layout(
    title={
        'text': "Average per Teams",
        'y':1,
        'x':0.45,
        'xanchor': 'center',
        'yanchor': 'top'})
fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig.update_layout({'plot_bgcolor': 'white',
                   'paper_bgcolor': 'white'})
fig.update_layout(width=1000, height=600,
                  margin = dict(t=15, l=15, r=15, b=15))
fig.show()
#6.msdhighscorevs teams
df = df.dropna()
sns.barplot(x='H.S', y='Teams', data=df1)
plt.xlabel('H.S')
plt.ylabel('Teams')
plt.title('Grouped bar plot of High score vs Teams')
plt.show()

#7.msdmatchin eachgrd
df = df.dropna()
sns.barplot(x='Matches', y='Ground', data=df2)
plt.xlabel('Matches')
plt.ylabel('Ground')
plt.title('Grouped bar plot of Matches vs Ground')
plt.show()

#8.msdaveragevs ground
Df_info=df.copy()

match_df = pd.concat([df['Average']]).value_counts().reset_index()
match_df.set_axis(['Average','Ground'],axis = 'columns' ,inplace = True)

def get_color(name, number):
    pal = list(sns.color_palette(palette=name, n_colors=number).as_hex())
    return pal

pal_vi = get_color('viridis_r', len(match_df))

fig = px.bar(df, x='Average', y='Ground', text='Average',
             color ='Average', color_discrete_sequence=pal_vi)
fig.update_layout(
    title={
        'text': "Average per Ground",
        'y':1,
        'x':0.45,
        'xanchor': 'center',
        'yanchor': 'top'})
fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig.update_layout({'plot_bgcolor': 'white',
                   'paper_bgcolor': 'white'})
fig.update_layout(width=1000, height=600,
                  margin = dict(t=15, l=15, r=15, b=15))
fig.show()
#9.msdstrikeratevs ground
Df_info=df.copy()

match_df = pd.concat([df['Strike Rate']]).value_counts().reset_index()
match_df.set_axis(['Strike Rate','Ground'],axis = 'columns' ,inplace = True)

def get_color(name, number):
    pal = list(sns.color_palette(palette=name, n_colors=number).as_hex())
    return pal

pal_vi = get_color('viridis_r', len(match_df))

fig = px.bar(df, x='Strike Rate', y='Ground', text='Strike Rate',
             color ='Strike Rate', color_discrete_sequence=pal_vi)
fig.update_layout(
    title={
        'text': "Strike Rate per Ground",
        'y':1,
        'x':0.45,
        'xanchor': 'center',
        'yanchor': 'top'})
fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig.update_layout({'plot_bgcolor': 'white',
                   'paper_bgcolor': 'white'})
fig.update_layout(width=1000, height=600,
             
#10.msdaveragevsyear
Df_info=df.copy()

match_df = pd.concat([df['Average']]).value_counts().reset_index()
match_df.set_axis(['Average','Year'],axis = 'columns' ,inplace = True)

def get_color(name, number):
    pal = list(sns.color_palette(palette=name, n_colors=number).as_hex())
    return pal

pal_vi = get_color('viridis_r', len(match_df))

fig = px.bar(df, x='Average', y='Year', text='Average',
             color ='Average', color_discrete_sequence=pal_vi)
fig.update_layout(
    title={
        'text': "Average per Year",
        'y':1,
        'x':0.45,
        'xanchor': 'center',
        'yanchor': 'top'})
fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig.update_layout({'plot_bgcolor': 'white',
                   'paper_bgcolor': 'white'})
fig.update_layout(width=1000, height=600,
                  margin = dict(t=15, l=15, r=15, b=15))
fig.show()
#11.msdstrikeragevsyear
Df_info=df.copy()

match_df = pd.concat([df['Strike Rate']]).value_counts().reset_index()
match_df.set_axis(['Strike Rate','Year'],axis = 'columns' ,inplace = True)

def get_color(name, number):
    pal = list(sns.color_palette(palette=name, n_colors=number).as_hex())
    return pal

pal_vi = get_color('viridis_r', len(match_df))

fig = px.bar(df, x='Strike Rate', y='Year', text='Strike Rate',
             color ='Strike Rate', color_discrete_sequence=pal_vi)
fig.update_layout(
    title={
        'text': "Strike Rate per Year",
        'y':1,
        'x':0.45,
        'xanchor': 'center',
        'yanchor': 'top'})
fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig.update_layout({'plot_bgcolor': 'white',
                   'paper_bgcolor': 'white'})
fig.update_layout(width=1000, height=600,
                  margin = dict(t=15, l=15, r=15, b=15))
fig.show()
#12.msdfiftiesvsyear
Df_info=df.copy()

match_df = pd.concat([df['Fifties']]).value_counts().reset_index()
match_df.set_axis(['Fifties','Year'],axis = 'columns' ,inplace = True)

def get_color(name, number):
    pal = list(sns.color_palette(palette=name, n_colors=number).as_hex())
    return pal

pal_vi = get_color('viridis_r', len(match_df))

fig = px.bar(df, x='Fifties', y='Year', text='Fifties',
             color ='Fifties', color_discrete_sequence=pal_vi)
fig.update_layout(
    title={
        'text': "Fifties per Year",
        'y':1,
        'x':0.45,
        'xanchor': 'center',
        'yanchor': 'top'})
fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig.update_layout({'plot_bgcolor': 'white',
                   'paper_bgcolor': 'white'})
fig.update_layout(width=1000, height=600,
                  margin = dict(t=15, l=15, r=15, b=15))
fig.show()
#####

#1.dkmatchesvs teams
sns.barplot(x='Matches', y='Teams', data=df3)
plt.xlabel('Matches')
plt.ylabel('Teams')
plt.title('Grouped bar plot of Matches vs Teams')
plt.show()

#2.dkinningssvs teams
df = df.dropna()
sns.barplot(x='Innings', y='Teams', data=df3)
plt.xlabel('Innings')
plt.ylabel('Teams')
plt.title('Grouped bar plot of Innings vs Teams')
plt.show()

#3.dkrunssvs teams
df = df.dropna()
sns.barplot(x='Runs', y='Teams', data=df3)
plt.xlabel('Runs')
plt.ylabel('Teams')
plt.title('Grouped bar plot of Runs vs Teams')
plt.show()
#4.dkstrikeratevs teams
Df_info=df.copy()

match_df = pd.concat([df['Strike Rate']]).value_counts().reset_index()
match_df.set_axis(['Strike Rate','Teams'],axis = 'columns' ,inplace = True)

def get_color(name, number):
    pal = list(sns.color_palette(palette=name, n_colors=number).as_hex())
    return pal

pal_vi = get_color('viridis_r', len(match_df))

fig = px.bar(df, x='Strike Rate', y='Teams', text='Strike Rate',
             color ='Strike Rate', color_discrete_sequence=pal_vi)
fig.update_layout(
    title={
        'text': "Strike Rate per Teams",
        'y':1,
        'x':0.45,
        'xanchor': 'center',
        'yanchor': 'top'})
fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig.update_layout({'plot_bgcolor': 'white',
                   'paper_bgcolor': 'white'})
fig.update_layout(width=1000, height=600,
             
#5.dkaveragevs teams
Df_info=df.copy()

match_df = pd.concat([df['Average']]).value_counts().reset_index()
match_df.set_axis(['Average','Teams'],axis = 'columns' ,inplace = True)

def get_color(name, number):
    pal = list(sns.color_palette(palette=name, n_colors=number).as_hex())
    return pal

pal_vi = get_color('viridis_r', len(match_df))

fig = px.bar(df, x='Average', y='Teams', text='Average',
             color ='Average', color_discrete_sequence=pal_vi)
fig.update_layout(
    title={
        'text': "Average per Teams",
        'y':1,
        'x':0.45,
        'xanchor': 'center',
        'yanchor': 'top'})
fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig.update_layout({'plot_bgcolor': 'white',
                   'paper_bgcolor': 'white'})
fig.update_layout(width=1000, height=600,
                  margin = dict(t=15, l=15, r=15, b=15))
fig.show()
#6.dkhighscorevs teams
df = df.dropna()
sns.barplot(x='H.S', y='Teams', data=df3)
plt.xlabel('H.S')
plt.ylabel('Teams')
plt.title('Grouped bar plot of High score vs Teams')
plt.show()
#7.dkmatchin eachgrd
df = df.dropna()
sns.barplot(x='Matches', y='Ground', data=df4)
plt.xlabel('Matches')
plt.ylabel('Ground')
plt.title('Grouped bar plot of Matches vs Ground')
plt.show()
#8.dkaveragevs ground
Df_info=df.copy()

match_df = pd.concat([df['Average']]).value_counts().reset_index()
match_df.set_axis(['Average','Ground'],axis = 'columns' ,inplace = True)

def get_color(name, number):
    pal = list(sns.color_palette(palette=name, n_colors=number).as_hex())
    return pal

pal_vi = get_color('viridis_r', len(match_df))

fig = px.bar(df, x='Average', y='Ground', text='Average',
             color ='Average', color_discrete_sequence=pal_vi)
fig.update_layout(
    title={
        'text': "Average per Ground",
        'y':1,
        'x':0.45,
        'xanchor': 'center',
        'yanchor': 'top'})
fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig.update_layout({'plot_bgcolor': 'white',
                   'paper_bgcolor': 'white'})
fig.update_layout(width=1000, height=600,
                  margin = dict(t=15, l=15, r=15, b=15))
fig.show()
#9.dkstrikeratevs ground
Df_info=df.copy()

match_df = pd.concat([df['Strike Rate']]).value_counts().reset_index()
match_df.set_axis(['Strike Rate','Ground'],axis = 'columns' ,inplace = True)

def get_color(name, number):
    pal = list(sns.color_palette(palette=name, n_colors=number).as_hex())
    return pal

pal_vi = get_color('viridis_r', len(match_df))

fig = px.bar(df, x='Strike Rate', y='Ground', text='Strike Rate',
             color ='Strike Rate', color_discrete_sequence=pal_vi)
fig.update_layout(
    title={
        'text': "Strike Rate per Ground",
        'y':1,
        'x':0.45,
        'xanchor': 'center',
        'yanchor': 'top'})
fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig.update_layout({'plot_bgcolor': 'white',
                   'paper_bgcolor': 'white'})
fig.update_layout(width=1000, height=600,
             
#10.dkaveragevsyear
Df_info=df.copy()

match_df = pd.concat([df['Average']]).value_counts().reset_index()
match_df.set_axis(['Average','Year'],axis = 'columns' ,inplace = True)

def get_color(name, number):
    pal = list(sns.color_palette(palette=name, n_colors=number).as_hex())
    return pal

pal_vi = get_color('viridis_r', len(match_df))

fig = px.bar(df, x='Average', y='Year', text='Average',
             color ='Average', color_discrete_sequence=pal_vi)
fig.update_layout(
    title={
        'text': "Average per Year",
        'y':1,
        'x':0.45,
        'xanchor': 'center',
        'yanchor': 'top'})
fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig.update_layout({'plot_bgcolor': 'white',
                   'paper_bgcolor': 'white'})
fig.update_layout(width=1000, height=600,
                  margin = dict(t=15, l=15, r=15, b=15))
fig.show()
#11.dkstrikeragevsyear
Df_info=df.copy()

match_df = pd.concat([df['Strike Rate']]).value_counts().reset_index()
match_df.set_axis(['Strike Rate','Year'],axis = 'columns' ,inplace = True)

def get_color(name, number):
    pal = list(sns.color_palette(palette=name, n_colors=number).as_hex())
    return pal

pal_vi = get_color('viridis_r', len(match_df))

fig = px.bar(df, x='Strike Rate', y='Year', text='Strike Rate',
             color ='Strike Rate', color_discrete_sequence=pal_vi)
fig.update_layout(
    title={
        'text': "Strike Rate per Year",
        'y':1,
        'x':0.45,
        'xanchor': 'center',
        'yanchor': 'top'})
fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig.update_layout({'plot_bgcolor': 'white',
                   'paper_bgcolor': 'white'})
fig.update_layout(width=1000, height=600,
                  margin = dict(t=15, l=15, r=15, b=15))
fig.show()
#12.dkfiftiesvsyear
Df_info=df.copy()

match_df = pd.concat([df['Fifties']]).value_counts().reset_index()
match_df.set_axis(['Fifties','Year'],axis = 'columns' ,inplace = True)

def get_color(name, number):
    pal = list(sns.color_palette(palette=name, n_colors=number).as_hex())
    return pal

pal_vi = get_color('viridis_r', len(match_df))

fig = px.bar(df, x='Fifties', y='Year', text='Fifties',
             color ='Fifties', color_discrete_sequence=pal_vi)
fig.update_layout(
    title={
        'text': "Fifties per Year",
        'y':1,
        'x':0.45,
        'xanchor': 'center',
        'yanchor': 'top'})
fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig.update_layout({'plot_bgcolor': 'white',
                   'paper_bgcolor': 'white'})
fig.update_layout(width=1000, height=600,
                  margin = dict(t=15, l=15, r=15, b=15))
fig.show()
#bounariescomparison
#msd
sixes= df['Sixes'].sum()
fours = df['Fours'].sum()
data = go.Pie(labels=['Sixes', 'Fours'], values=[sixes, fours])
layout = go.Layout(title='Percentage of sixes vs fours')
fig = go.Figure(data=[data], layout=layout)
fig.show()

#dk
sixes= df['Sixes'].sum()
fours = df['Fours'].sum()
data = go.Pie(labels=['Sixes', 'Fours'], values=[sixes, fours])
layout = go.Layout(title='Percentage of sixes vs fours')
fig = go.Figure(data=[data], layout=layout)
fig.show()

#inninigscomparison

#comparison on average
trace1 = go.Bar(x=df['Year'], y=df['MSD Average'], name='DK Average')
trace2 = go.Bar(x=df['Year'], y=df['MSD Average'], name='DK Average')
layout = go.Layout(title='Average by Year', xaxis_title='Year', yaxis_title='Average')
fig = go.Figure(data=[trace1,trace2], layout=layout)
fig.show()

#comparison on strikerate
trace1 = go.Bar(x=df['Year'], y=df['MSD Strike Rate'], name='DK Strike Rate')
trace2 = go.Bar(x=df['Year'], y=df['MSD Strike Rate'], name='DK Strike Rate')
layout = go.Layout(title='Strike Rate by Year', xaxis_title='Year', yaxis_title='Strike Rate')
fig = go.Figure(data=[trace1,trace2], layout=layout)
fig.show()

#comparison on msdgettingout
out_method = [" "]
out_times = []
fig = go.Figure(data=[go.Pie(labels=out_method, values=out_times)])
fig.update_layout(title="comparison on msdgettingout")
fig.show()
#comparison on dkgettingout
out_method = [" "]
out_times = []
fig = go.Figure(data=[go.Pie(labels=out_method, values=out_times)])
fig.update_layout(title="comparison on dkgettingout")
fig.show()

#performancebybattingposition
#msdavg
sns.barplot(x='Batting Position', y='Average', data=df1)
plt.xlabel('Batting Position')
plt.ylabel('Average')
plt.title('Average in Batting Position')
plt.show()

#msdsr
sns.barplot(x='Batting Position', y='Strike Rate', data=df1)
plt.xlabel('Batting Position')
plt.ylabel('Strike Rate')
plt.title('Strike Rate in Batting Position')
plt.show()

#dkavg
sns.barplot(x='Batting Position', y='Average', data=df1)
plt.xlabel('Batting Position')
plt.ylabel('Average')
plt.title('Average in Batting Position')
plt.show()

#dksr
sns.barplot(x='Batting Position', y='Strike Rate', data=df1)
plt.xlabel('Batting Position')
plt.ylabel('Strike Rate')
plt.title('Strike Rate in Batting Position')
plt.show()

#matchwinningknocksofMSd

#ratio of resulting runs
batting_positions = ["Runs% at wins","Runs% at Loss"]
Team_Runs = []
fig = go.Figure(data=[go.Pie(labels=batting_positions, values=Team_Runs)])
fig.update_layout(title="Performance by Match Result")
fig.show()
#strike RATEINIPLCAREERMSD
sns.barplot(x='RESULT', y='Strike Rate', data=df1)
plt.xlabel('RESULT')
plt.ylabel('Strike Rate')
plt.title('Strike Rate in Match Result')
plt.show()

#matchwinningknocksofdk

#ratio of resulting runs
batting_positions = ["Runs% at wins","Runs% at Loss"]
Team_Runs = [2170,2179]
fig = go.Figure(data=[go.Pie(labels=batting_positions, values=Team_Runs)])
fig.update_layout(title="Performance by Match Result")
fig.show()
#strike RATEINIPLCAREERDK
sns.barplot(x='RESULT', y='Strike Rate', data=df1)
plt.xlabel('RESULT')
plt.ylabel('Strike Rate')
plt.title('Strike Rate in Match Result')
plt.show()

#MSDWKPER
trace1 = go.Bar(x=df['Year'], y=df['Catches Taken'], name='Catches Taken')
trace2 = go.Bar(x=df['Year'], y=df['Stumpings'], name='Stumpings')
layout = go.Layout(title='Wicketkeeping Performance', xaxis_title='Year', yaxis_title='Catches/Stumping')
fig = go.Figure(data=[trace1,trace2], layout=layout)
fig.show()
#DKWKPER
trace1 = go.Bar(x=df['Year'], y=df['Catches Taken'], name='Catches Taken')
trace2 = go.Bar(x=df['Year'], y=df['Stumpings'], name='Stumpings')
layout = go.Layout(title='Wicketkeeping Performance', xaxis_title='Year', yaxis_title='Catches/Stumping')
fig = go.Figure(data=[trace1,trace2], layout=layout)
fig.show()
