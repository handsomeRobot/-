
# coding: utf-8

# In[1]:





# In[2]:


import pandas as pd
import numpy as np


# In[3]:


def entropy(data, feature):
    entropy = 0
    if feature != "":
        branches = list(set(data[feature]))
        for br in branches:
            br_data = data[data[feature] == br]
            br_length = len(br_data)
            p_yes = len(br_data[br_data.label == 'yes']) * 1.0 / br_length
            p_no = len(br_data[br_data.label == 'no']) * 1.0 / br_length
            p_yes_eff, p_no_eff = p_yes, p_no
            if not p_yes:
                p_yes_eff = 1
            if not p_no:
                p_no_eff = 1
            entropy += br_length * 1.0 / len(data) * (-p_yes * np.log(p_yes_eff) - p_no * np.log(p_no_eff))
        return entropy
    elif feature == "":
        p_yes = len(data[data.label == 'yes']) * 1.0 / len(data)
        p_no = len(data[data.label == 'no']) * 1.0 / len(data)
        entropy = -p_yes * np.log(p_yes) - p_no * np.log(p_no)
        return entropy


# In[4]:


data = pd.DataFrame({"age": ['young', 'young', 'young', 'young', 'young', 'middle', 'middle', 'middle', 'middle', 'middle', 'old', 'old', 'old', 'old', 'old'], 
                     "haveWork": ['no', 'no', 'yes', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no', 'no', 'no', 'yes', 'yes', 'no'],
                     "haveHouse": ['no', 'no', 'no', 'yes', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no', 'no', 'no'], 
                     "credit": ['normal', 'good', 'good', 'normal', 'normal', 'normal', 'good', 'good', 'very good', 'very good', 'very good', 
                               'good', 'good', 'very good', 'normal'], 
                     "label": ['no', 'no', 'yes', 'yes', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']})
data


# In[5]:


(entropy(data, "") - entropy(data, "age")) / entropy(data, '')


# In[6]:


(entropy(data, "") - entropy(data, "credit")) / entropy(data, '')


# In[7]:


(entropy(data, "") - entropy(data, "haveHouse")) / entropy(data, '')


# In[8]:


(entropy(data, "") - entropy(data, "haveWork")) / entropy(data, '')

