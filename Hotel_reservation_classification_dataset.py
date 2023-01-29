#!/usr/bin/env python
# coding: utf-8

# In[49]:


import pandas as pd
import numpy as np

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow.keras import layers


# In[4]:


get_ipython().system('pip install kaggle')
#titanic = pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/train.csv")
#itanic.head()


# In[6]:


import kaggle #This imports the kaggle json file located in C:\Users\prasa\.kaggle 


# In[7]:


from kaggle.api.kaggle_api_extended import KaggleApi #Importing our Api


# In[8]:


#Initializing and authenticating our Api

api = KaggleApi()
api.authenticate()


# In[9]:


#This is how to download competition dataset

api.competition_download_file('spaceship-titanic','train.csv')


# In[11]:


#Now to download stand alone data. 

test1 = api.dataset_download_file('ahsan81/hotel-reservations-classification-dataset','Hotel Reservations.csv')


# Note: If there is a zip file. 
# 
# `import zipfile
# with zipfile.Zipfile('filename.tsv.zip','r') as zipref:
# zipref.extractall()
# `

# In[47]:



#test1 = pd.read_csv('C:\\Users\\prasa\\Downloads\\Kaggle Files\\Hotel_Reservations_Dataset\\Hotel Reservations.csv',sep ='\t')
#Note: The code below has been working because of the UTF-8. It has rectified the encoding.
df  = pd.read_csv('C:\\Users\\prasa\\Downloads\\Kaggle Files\\Hotel_Reservations_Dataset\\Hotel Reservations.csv',encoding='UTF-8')
print(df.iloc[:,2:])


# # Question: Can you predict if the customer is going to honor the reservation or cancel it ?
# 
# Logic: If a person have `no_of_previous_cancellations`

# In[58]:


import pandas as pd
import numpy as np

msk = np.random.rand(len(df)) <= 0.7

train = df[msk]
test = df[~msk]


# In[62]:


print(train.shape)
print(test.shape)
print(df.shape)


# In[ ]:





# In[ ]:





# In[ ]:




