#!/usr/bin/env python
# coding: utf-8

# ### Importing libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# ### Importing the dataset

# In[2]:


dt=pd.read_csv("D:\projects\EDA\Heart_Failure\data\heart.csv")
dt.head(2)


# ### Dataset Information

# Names of all the Features present in the dataset

# In[3]:


dt.columns


# About the columns and the datatypes of values they contains

# In[4]:


dt.info()


# In the data, we can see that columns "age" and "platelets" contains integral values but have datatype as float, hence changing its data type to integer.

# In[5]:


dt["platelets"]=dt["platelets"].astype(int)
dt["age"]=dt["age"].astype(int)


# In[6]:


dt.info()


# Here, we can see that datatype of "age" and "platelets" has been changed to integer.

# Statistical information about the columns

# In[7]:


dt.describe()


# ### Handling Missing values

# In[8]:


dt[dt.isnull()==True].count()


# There are no missing or null values in any of the column.

# ### Removing the duplicate rows

# In[9]:


dt.drop_duplicates(inplace=True)


# In[10]:


yes = dt[dt['DEATH_EVENT'] == 1].describe().transpose()
no = dt[dt['DEATH_EVENT'] == 0].describe().transpose()


# In[11]:


yes


# # Exploratory Data Analysis

# In[12]:


fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))

plt.subplot(1,2,1)
sns.heatmap(yes[['mean']],annot = True,cmap = "Blues",linewidths = 1,linecolor = 'black',cbar = False,fmt = '.2f')
plt.title("Death Occured")

plt.subplot(1,2,2)
sns.heatmap(yes[['mean']],annot = True,cmap = "Blues",linewidths = 1,linecolor = 'black',cbar = True,fmt = '.2f')
plt.title("No Death Occured")

fig.tight_layout()
plt.show()


# ### Dividing the Features to Categorical Features and Numerical Features

# We have dividing the features on the basis that if the column has more than 5 unique values then it is categorised as Numerical feature and if it has less aur equal to 5 unique values then it is categorised as Categorical feature.

# In[13]:


Categorical=[]
Numerical=[]

for x in dt.columns:
    if len(dt[x].unique())<=5:
        Categorical.append(x)
    else:
        Numerical.append(x)

c_len=len(Categorical)
n_len=len(Numerical)

print("Categical_featues:",Categorical,"\n","Numerical_features:",Numerical)


# ## Plotting Distibution of Categorical Features

# In[14]:


fig, ax = plt.subplots(nrows = 3,ncols = 2,figsize = (10,15))

for x in range(c_len):
    plt.subplot(3,2,x+1)
    sns.distplot(dt[Categorical[x]],kde_kws = {'bw' : 1},color="cornflowerblue")
    plt.title(Categorical[x]+" Distribution")
    
plt.suptitle("Distribution of Categorical Features\n",fontsize=15)    
fig.tight_layout(h_pad=2,w_pad=2)
plt.show()


# From the Distribution curves of all the Categorical features, we can say that all Categorical features are normally distributed.

# ## Plotting Distibution of Numerical Features

# In[15]:


fig, ax = plt.subplots(nrows = 3,ncols = 2,figsize = (10,15))
plt.suptitle("Distribution of Numerical Features\n",fontsize=15)    

for x in range(n_len-1):
    plt.subplot(3,2,x+1)
    sns.distplot(dt[Numerical[x]],kde_kws = {'bw' : 0.5},color="cornflowerblue")
    plt.title(Numerical[x]+" Distribution")

plt.figure(figsize=(5,5))
sns.distplot(dt[Numerical[n_len-1]],kde_kws = {'bw' : 0.5},color="cornflowerblue")
plt.title(Numerical[n_len-1]+" Distribution")

fig.tight_layout(h_pad=2,w_pad=2)
plt.show()


# From the above graphs, we can say that Age,Creatinine_Phosphokinase,Ejaction_Fraction and Serum_creatinine data Distribution graphs have a longer tail on the right side of the peak and Platelets and Serum_Sodium data have almost equal tails on both side of the peak whereas Time data Distribution is a typical time series graph. Hence,
# - Age,Creatinine_Phosphokinase,Ejaction_Fraction and Serum_creatinine - **Rightly or Positively skewed data distribution**
# - Platelets and Serum_Sodium - **Normally distributed**
# - Time - **Time Series Graph**
# 

# ## Visualizing theTarget Value (DEATH_EVENT)

# In[16]:


t_data = dt['DEATH_EVENT'].value_counts()

fig=plt.subplots(ncols=2,figsize=(20,5))

plt.subplot(1,2,1)
plt.bar(["No Death Occurred","Death Ocuurred"],t_data,color=["white","cornflowerblue"],edgecolor="black")
for i in range(2):
        plt.text(i,t_data[i]+2, t_data[i], ha = 'center')
plt.title("Death Count of Death Cases")
plt.ylabel("Counts")
plt.xlabel("Death Case")


plt.subplot(1,2,2)
plt.pie(t_data,labels=["No Death Occurred","Death Ocuurred"],startangle=90,autopct="%.2f%%",pctdistance=0.5,explode=[0,0.1],colors=["white","cornflowerblue"], wedgeprops = {'edgecolor' : 'black','linewidth': 1,'antialiased' : True})
plt.title("Death Percentage of Death Cases")
        
plt.tight_layout()
plt.show()


# We can see that both categories are uneaqual and are almost in the ratio of 2:1 where No Death Cases are almost as twice as Death Cases.
# - No Death Cases : Death Cases = 2 : 1
# - Dataset is unbalanced
# Since, the dataset is unbalanced the prediction will biased towards the category with data. In this case, it No Death Case.

# ## Plotting Categorical Features vs Target Feature

# In[17]:


fig=plt.subplots(nrows=3,ncols=2,figsize=(15,10))
for i in range(c_len-2):
    plt.subplot(2,2,i+1)
    axs=sns.countplot(x=Categorical[i],data = dt,hue = "DEATH_EVENT",edgecolor = 'black',color="cornflowerblue")
    for rect in axs.patches:
        axs.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 2, rect.get_height(), horizontalalignment='center', fontsize = 11)
    axs.set_xticklabels([Categorical[i]+"=0",Categorical[i]+"=1"])
    plt.title(Categorical[i] + ' vs DEATH_EVENT');

plt.figure(figsize=(5,3.5))
axs=sns.countplot(x=Categorical[4],data = dt,hue = "DEATH_EVENT",edgecolor = 'black',color="cornflowerblue")
for rect in axs.patches:
        axs.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 2, rect.get_height(), horizontalalignment='center', fontsize = 8)
axs.set_xticklabels([Categorical[4]+"=0",Categorical[4]+"=1"])
plt.title(Categorical[4] + ' vs DEATH_EVENT');

plt.tight_layout()
plt.show()


# From the above graphs, we can say that patients without anaemia, diabetes, high blood pressure, and smoking habits have higher death rates than those with these conditions. Additionally, more males than females experience death due to heart failure.

# ## Categorical Features vs Cases of Death in Heart Failure

# In[18]:


ax,fig = plt.subplots(nrows = 1,ncols = 3,figsize = (20,15))

for i in range(3):
    plt.subplot(1,3,i+1)
    plt.pie(dt[dt['DEATH_EVENT'] == 1][Categorical[i]].value_counts(),labels = ['No '+Categorical[i],Categorical[i]],autopct='%.2f%%',startangle = 90,explode = (0,0.1),wedgeprops = {'edgecolor' : 'black','linewidth': 1,'antialiased' : True},colors=["white","cornflowerblue"])
    plt.title(Categorical[i]);
    

ax,fig = plt.subplots(nrows = 1,ncols = 2,figsize = (10,10))
    
plt.subplot(1,2,1)
plt.pie(dt[dt['DEATH_EVENT'] == 1]["sex"].value_counts().sort_index(),labels = ["Female","Male"],autopct='%.2f%%',startangle = 90,explode = (0,0.1),wedgeprops = {'edgecolor' : 'black','linewidth': 1,'antialiased' : True},colors=["white","cornflowerblue"])
plt.title("Sex");

plt.subplot(1,2,2)
plt.pie(dt[dt['DEATH_EVENT'] == 1]["smoking"].value_counts(),labels = ["No Smoking","Smoking"],autopct='%.2f%%',startangle = 90,explode = (0,0.1),wedgeprops = {'edgecolor' : 'black','linewidth': 1,'antialiased' : True},colors=["white","cornflowerblue"])
plt.title("Smoking");

plt.tight_layout()
plt.show()


# The above pie charts indicate that patients without anaemia, diabetes, high blood pressure, and smoking habits have more death events than those with these conditions. Additionally, males are more prone to heart failure-related death events than females.

# ## Plotting Numerical Features vs Target Feature

# First we will Scale the creatinine_phosphokinase, platelets and time columns as these columns contains a wide range of values which will be hard interpret if graphed directly.

# In[19]:


dt['creatinine_phosphokinase_scaled'] = [ int(i / 100) for i in dt['creatinine_phosphokinase']]
dt['platelets_scaled'] = [ int(i / 10**5) for i in dt['platelets']]
dt['time_scaled'] = [ int(i / 5) for i in dt['time']]


# In[20]:


Numerical_scaled=Numerical
Numerical_scaled[1]='creatinine_phosphokinase_scaled'
Numerical_scaled[3]='platelets_scaled'
Numerical_scaled[6]='time_scaled'
ns_len=len(Numerical_scaled)


# In[21]:


fig, ax = plt.subplots(nrows = 7,ncols = 1,figsize = (11,30),squeeze = False)

for i in range(ns_len):
    plt.subplot(7,1,i+1)
    sns.countplot(x=Numerical_scaled[i],data = dt,hue = "DEATH_EVENT",color="cornflowerblue",edgecolor = 'black')
    plt.title("\n"+Numerical_scaled[i] + ' vs DEATH EVENT');
    
plt.tight_layout()
plt.show()


# - Death cases begin at age 45 and show peaks at ages 45, 50, 60, 65, 70, 75, and 80. 
# - Higher Death cases are observed for creatinine_phosphokinase values between (0x100)-(5x100) i.e. 0-500.
# - High Death rates are associated with ejection fraction values of 20-60.
# - High Death cases are observed for platelets values between (0x10^5)-(4x10^5) i.e. 0-400,000.
# - High Death rates are associated with serum creatinine levels of 0.6-3.0.
# - High Death cases are observed for serum sodium levels of 127-145.
# - time values from 0-170 show a higher probability of leading to a Death.

# ## Categorical features vs Numerical features with respect to Target feature

# #### Anaemia vs Numerical Features

# In[22]:


fig=plt.subplots(1,3,figsize=(15,5))
for i in range(3):
    plt.subplot(1,3,i+1)
    sns.boxplot(x="anaemia",y = Numerical[i],data=dt,hue = 'DEATH_EVENT',palette = ["white","cornflowerblue"])
    plt.title('Anaemia vs '+Numerical[i])

plt.suptitle("Anaemia vs Numerical features",fontsize=20)
plt.tight_layout()
    
fig=plt.subplots(2,2,figsize=(10,10))
for i in range(4):
    plt.subplot(2,2,i+1)
    sns.boxplot(x="anaemia",y = Numerical[i+3],data=dt,hue = 'DEATH_EVENT',palette = ["white","cornflowerblue"])
    plt.title('Anaemia vs '+Numerical[i+3])
    
plt.tight_layout()
plt.show()


# Individuals aged 55-75 and with ejection fraction values of 20-40 are prone to Death due to Heart Failure, regardless of anaemia. Additionally, serum_creatinine levels between 1-2 and serum_sodium levels of 130-140 show a higher risk of Death.

# #### Diabetes vs Numerical Features

# In[23]:


fig=plt.subplots(1,3,figsize=(15,5))
for i in range(3):
    plt.subplot(1,3,i+1)
    sns.boxplot(x="diabetes",y = Numerical[i],data=dt,hue = 'DEATH_EVENT',palette = ["white","cornflowerblue"])
    plt.title('Diabetes vs '+Numerical[i])

plt.suptitle("Diabetes vs Numerical features",fontsize=20)
plt.tight_layout()
    
fig=plt.subplots(2,2,figsize=(10,10))
for i in range(4):
    plt.subplot(2,2,i+1)
    sns.boxplot(x="diabetes",y = Numerical[i+3],data=dt,hue = 'DEATH_EVENT',palette = ["white","cornflowerblue"])
    plt.title('Diabetes vs '+Numerical[i+3])
    
plt.tight_layout()
plt.show()


# Higher heart failure cases are observed for creatinine_phosphokinase values from 0-500 and platelets ranging from 200,000-300,000. Additionally, serum_creatinine levels between 1-2 and time values from 0-100 also indicate more heart failure cases.

# #### High Blood Pressure vs Numerical Features

# In[24]:


fig=plt.subplots(1,3,figsize=(15,5))
for i in range(3):
    plt.subplot(1,3,i+1)
    sns.boxplot(x="high_blood_pressure",y = Numerical[i],data=dt,hue = 'DEATH_EVENT',palette = ["white","cornflowerblue"])
    plt.title('High blood pressure vs '+Numerical[i])

plt.suptitle("High blood pressure vs Numerical features",fontsize=20)
plt.tight_layout()
    
fig=plt.subplots(2,2,figsize=(10,10))
for i in range(4):
    plt.subplot(2,2,i+1)
    sns.boxplot(x="high_blood_pressure",y = Numerical[i+3],data=dt,hue = 'DEATH_EVENT',palette = ["white","cornflowerblue"])
    plt.title('High blood pressure vs '+Numerical[i+3])
    
plt.tight_layout()
plt.show()


# High blood pressure extends the age range of Death due to heart failure, lowering the lower age limit below 55 and raising the upper limit above 70. It also reduces the time feature's values, increasing the likelihood of heart failure.

# #### Sex vs Numerical Features

# In[25]:


fig=plt.subplots(1,3,figsize=(15,5))
for i in range(3):
    plt.subplot(1,3,i+1)
    sns.boxplot(x="sex",y = Numerical[i],data=dt,hue = 'DEATH_EVENT',palette = ["white","cornflowerblue"])
    plt.title('Sex vs '+Numerical[i])

plt.suptitle("Sex vs Numerical features",fontsize=20)
plt.tight_layout()
    
fig=plt.subplots(2,2,figsize=(10,10))
for i in range(4):
    plt.subplot(2,2,i+1)
    sns.boxplot(x='sex',y = Numerical[i+3],data=dt,hue = 'DEATH_EVENT',palette = ["white","cornflowerblue"])
    plt.title('Sex vs '+Numerical[i+3])
    
plt.tight_layout()
plt.show()


# Females aged 50-70 and males aged 60-75 are more prone to heart failure leading to Death. Ejection fraction values of 30-50 for females and 20-40 for males are associated with DEATH_EVENT. Serum sodium values indicating Death case due to heart failure vary between males and females.

# #### Smoking vs Numerical Features

# In[26]:


fig=plt.subplots(1,3,figsize=(15,5))
for i in range(3):
    plt.subplot(1,3,i+1)
    sns.boxplot(x="smoking",y = Numerical[i],data=dt,hue = 'DEATH_EVENT',palette = ["white","cornflowerblue"])
    plt.title('Smoking vs '+Numerical[i])

plt.suptitle("Smoking vs Numerical features",fontsize=20)
plt.tight_layout()
    
fig=plt.subplots(2,2,figsize=(10,10))
for i in range(4):
    plt.subplot(2,2,i+1)
    sns.boxplot(x='smoking',y = Numerical[i+3],data=dt,hue = 'DEATH_EVENT',palette = ["white","cornflowerblue"])
    plt.title('Smoking vs '+Numerical[i+3])
    
plt.tight_layout()
plt.show()


# For Death due to smoking, the age group 60-70 dominates. For non-smokers, the age range for DEATH_EVENT cases increases to 50-75. Smoking reduces the range of the time feature for Death to 0-75.

# ## Numerical features vs Numerical features with respect to Target feature

# In[27]:


l=0
fig = plt.subplots(nrows = 11,ncols = 2,figsize = (20,55),squeeze = False)
for i in range(ns_len - 1):
    for j in range(ns_len):
        if i != j and j > i:
            l += 1
            plt.subplot(11,2,l)
            sns.scatterplot(x = Numerical[i],y = Numerical[j],data = dt,hue = 'DEATH_EVENT',palette = ["white","cornflowerblue"],edgecolor = 'black');
            plt.title(Numerical[i] + ' vs ' + Numerical[j])
            
plt.tight_layout()
plt.show()


# - Peaks in Death occurrences are observed at ages 50, 60, 70, and 80 within the time range of 50-100.
# - High creatinine_phosphokinase values (0-500) are prevalent in Death cases regardless of other factors.
# - Ejaction_fraction values from 20-40 show a high number of Death instances.
# - Platelets values ranging from 2x10^5 to 4x10^5 combined with time values from 0-50 are significant indicators for Death.
# - Serum_creatinine values between 0-2 with time values from 0-50 indicate a higher likelihood of Death.
# - Serum_sodium values from 130-140 are associated with a high number of Death cases.

# # Summary 

# #### When the Death due to Heart Failure Occured
# The dataset includes categorical features such as anaemia (yes/no), diabetes (yes/no), high blood pressure (yes/no, with additional data needed), sex (male/female), and smoking status (smoker/non-smoker). These variables offer insights into health conditions and lifestyle factors relevant to cardiovascular health.
# 
# Numerical features encompass age (50-70 years), creatinine phosphokinase levels (0-500), ejection fraction (20-40), platelet counts (200,000-300,000), serum creatinine levels (1-2), serum sodium levels (130-140), and time (0-50), likely representing duration of observation. These measurements provide quantitative data crucial for analyzing cardiovascular health and potential correlations with various health indicators.
# 
# 
# 
# 

# #### General Findings
# - Categorical features such as anaemia, diabetes, high blood pressure, male sex (slightly more vulnerable than females), and smoking are associated with increased risks of heart failure.
# 
# - Numerical factors including older age, elevated creatinine phosphokinase levels (>120 mcg/L), reduced ejection fraction (<55%), extreme platelet counts, elevated serum creatinine (0.8 - 1.7 mg/dL), high serum sodium (>130 mEq/L), and prolonged follow-up periods (>14 days) indicate higher likelihoods of heart failure.
# 
# - These insights, gathered from research and empirical data, inform exploratory data analysis (EDA) and feature selection processes.
# 
# - Discrepancies between EDA findings and established domain knowledge (e.g., anaemia, diabetes, smoking) may be due to the dataset's modest size (299 data points) and imbalance (2:1 ratio of No DEATH_EVENT to DEATH_EVENT).
# 
# - Such factors highlight the need for careful interpretation and validation of findings in medical research to ensure robust conclusions and reliable predictive models for clinical applications.
