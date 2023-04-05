#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[15]:


Df_info=df1.copy()

match_df = pd.concat([df1['Strike Rate']]).value_counts().reset_index()
match_df.set_axis(['Strike Rate','Teams'],axis = 'columns' ,inplace = True)

def get_color(name, number):
    pal = list(sns.color_palette(palette=name, n_colors=number).as_hex())
    return pal

pal_vi = get_color('viridis_r', len(match_df))

fig = px.bar(df1, x='Strike Rate', y='Teams', text='Strike Rate',
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
                  margin = dict(t=15, l=15, r=15, b=15))
fig.show()


# In[11]:


Df_info=df1.copy()

match_df = pd.concat([df1['Average']]).value_counts().reset_index()
match_df.set_axis(['Average','Teams'],axis = 'columns' ,inplace = True)

def get_color(name, number):
    pal = list(sns.color_palette(palette=name, n_colors=number).as_hex())
    return pal

pal_vi = get_color('viridis_r', len(match_df))

fig = px.bar(df1, x='Average', y='Teams', text='Average',
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


# In[12]:


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


# In[13]:


Df_info=df2.copy()

match_df = pd.concat([df2['Average']]).value_counts().reset_index()
match_df.set_axis(['Average','Ground'],axis = 'columns' ,inplace = True)

def get_color(name, number):
    pal = list(sns.color_palette(palette=name, n_colors=number).as_hex())
    return pal

pal_vi = get_color('viridis_r', len(match_df))

fig = px.bar(df2, x='Average', y='Ground', text='Average',
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


# In[17]:


Df_info=df.copy()

match_df = pd.concat([df2['Strike Rate']]).value_counts().reset_index()
match_df.set_axis(['Strike Rate','Ground'],axis = 'columns' ,inplace = True)

def get_color(name, number):
    pal = list(sns.color_palette(palette=name, n_colors=number).as_hex())
    return pal

pal_vi = get_color('viridis_r', len(match_df))

fig = px.bar(df2, x='Strike Rate', y='Ground', text='Strike Rate',
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
                  margin = dict(t=15, l=15, r=15, b=15))
fig.show()


# In[22]:


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


# In[23]:


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


# In[ ]:




