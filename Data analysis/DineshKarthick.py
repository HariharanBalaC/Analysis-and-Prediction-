#!/usr/bin/env python
# coding: utf-8

# In[23]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import plotly.graph_objs as go

df1 = pd.read_csv('Downloads\data\dkvsopp.csv')
df2 = pd.read_csv('Downloads\data\dk.csv')
df3 = pd.read_csv('Downloads\data\dkinground.csv')


# In[5]:


print(df1.head())
print(df1.info())
print(df1.describe())
df1 = df1.dropna()


# In[8]:


sns.set(rc={'figure.figsize':(8,10)})
sns.barplot(x='Matches', y='Teams', data=df1)
plt.xlabel('Matches')
plt.ylabel('Teams')
plt.title('Grouped Bar Plot of Oppenent Team vs Matches')
plt.show()


# In[10]:


trace1 = go.Bar(x=df1['Teams'], y=df1['N.O'], name='No of Not Out')
trace2 = go.Bar(x=df1['Teams'], y=df1['Runs'], name='Runs')
layout = go.Layout(title='NotOut and Runs', xaxis_title='Teams', yaxis_title='NO&Runs')

fig = go.Figure(data=[trace1,trace2], layout=layout)

fig.show()


# In[11]:


trace1 = go.Bar(x=df1['Teams'], y=df1['Innings'], name='Highest Score')
trace2 = go.Bar(x=df1['Teams'], y=df1['Strike Rate'], name='Strike Rate')

layout = go.Layout(title='Catches and Stumpings in Year', xaxis_title='Teams', yaxis_title='Number of Highest score\Strike rate')

fig = go.Figure(data=[trace1,trace2], layout=layout)

fig.show()


# In[12]:


sns.set(rc={'figure.figsize':(8,10)})
sns.barplot(x='Innings', y='Teams', data=df1)
plt.xlabel('Innings')
plt.ylabel('Team')
plt.title('Grouped Bar Plot of Oppenent Teams vs Innings Played')
plt.show()


# In[13]:


sns.set(rc={'figure.figsize':(8,10)})
sns.barplot(x='Runs', y='Teams', data=df1)
plt.xlabel('Runs')
plt.ylabel('Teams')
plt.title('Grouped Bar Plot of Oppenent team vs Runs scored')
plt.show()


# In[14]:


sns.set(rc={'figure.figsize':(8,10)})
sns.barplot(x='Strike Rate', y='Teams', data=df1)
plt.xlabel('Strike Rate')
plt.ylabel('Teams')
plt.title('Grouped Bar Plot of Oppenent Team vs Strike Rate')
plt.show()


# In[15]:


sns.set(rc={'figure.figsize':(8,10)})
sns.barplot(x='Average', y='Teams', data=df1)
plt.xlabel('Average')
plt.ylabel('Teams')
plt.title('Grouped Bar Plot of Average vs Series')
plt.show()


# In[16]:


sns.set(rc={'figure.figsize':(12,10)})
sns.barplot(y='N.O', x='Teams', data=df1)
plt.xlabel('Highest Score')
plt.ylabel('Teams')
plt.title('Grouped Bar Plot of Oppenent Team vs Highest Score')
plt.xticks(rotation='vertical')
plt.show()


# In[17]:


print(df2.head()) 
print(df2.info())  
print(df2.describe())
df2 = df2.dropna()


# In[18]:


sns.set(rc={'figure.figsize':(12,8)})
sns.barplot(x='Series', y='Average', data=df2)
#valuelabel(Series,Average) 
plt.xlabel('Series')
plt.ylabel('Average')
plt.title('Grouped Bar Plot of Average vs Series')
plt.show()


# In[19]:


trace1 = go.Bar(x=df2['Series'], y=df2['6'], name='Sixes')
trace2 = go.Bar(x=df2['Series'], y=df2['4'], name='Fours')
layout = go.Layout(title='Boundaries in Year', xaxis_title='Year', yaxis_title='Boundaries')

fig = go.Figure(data=[trace1,trace2], layout=layout)

fig.show()


# In[20]:


sns.barplot(x='Innings', y='N.O', data=df2)
plt.xlabel('Innings')
plt.ylabel('N.O')
plt.title('Grouped Bar Plot of Notout innings')
plt.show()


# In[21]:


trace1 = go.Bar(x=df2['Series'], y=df2['Catches Taken'], name='Catches')
trace2 = go.Bar(x=df2['Series'], y=df2['Stumpings'], name='Stumpings')

layout = go.Layout(title='Catches and Stumpings in Year', xaxis_title='Year', yaxis_title='Number of Catches/Stumpings')

fig = go.Figure(data=[trace1,trace2], layout=layout)

fig.show()


# In[22]:


import plotly.graph_objs as go

sixes= df2['6'].sum()
fours = df2['4'].sum()
data = go.Pie(labels=['Sixes', 'Fours'], values=[sixes,fours])
layout = go.Layout(title='Percentage of sixes vs fours')

fig = go.Figure(data=[data], layout=layout)

fig.show()


# In[24]:


print(df3.head())
print(df3.info())
print(df3.describe())
df3 = df3.dropna()


# In[26]:


sns.set(rc={'figure.figsize':(12,12)})
sns.barplot(x='Matches', y='Ground', data=df3)
#plt.figure(figsize=(20,12))
plt.xlabel('Matches')
plt.ylabel('Ground')
plt.title('Grouped Bar Plot of Ground vs Matches')
plt.show()


# In[27]:


trace1 = go.Bar(x=df3['Ground'], y=df3['Average'], name='Average')
trace2 = go.Bar(x=df3['Ground'], y=df3['Strike Rate'], name='Strike Rate')

layout = go.Layout(title='Average and Strike rate in Ground', xaxis_title='Ground', yaxis_title='Average\Strike rate')

fig = go.Figure(data=[trace1,trace2], layout=layout)

fig.show()


# In[28]:


sns.barplot(x='Average', y='Ground', data=df3)
#plt.figure(figsize=(20,12))
plt.xlabel('Average')
plt.ylabel('Ground')
plt.title('Grouped Bar Plot of Ground vs Average')
plt.show()


# In[30]:


sns.set(rc={'figure.figsize':(12,12)})
sns.barplot(x='Strike Rate', y='Ground', data=df3)
#plt.figure(figsize=(20,12))
plt.xlabel('Strike Rate')
plt.ylabel('Ground')
plt.title('Grouped Bar Plot of Ground vs Strikerate')
plt.show()


# In[31]:


df3=df3.pivot_table(index='Ground',columns='Strike Rate',values='N.O',aggfunc='mean')
sns.heatmap(df3)


# In[ ]:




