
# coding: utf-8

# In[88]:


# import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

import pickle

import json
import requests


# In[89]:


import plotly.plotly as py
import cufflinks as cf
cf.go_offline()

from lxml import html

from bs4 import BeautifulSoup

from collections import OrderedDict

from selenium import webdriver
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


# In[97]:


df = pd.read_excel('../WebScraping/ASHRAE-Kaggle_WebScraping.xlsx', sheet_name = 'discussion', index_col = 0)


# In[104]:


print(df.info())


# ## Clean dataframe

# In[101]:


# Look for and drop duplicate rows
boolean = any(df['id'].duplicated())

df.drop_duplicates(subset=['id'], inplace=True)

print(df.info())


# ## Exploratory Data Analysis

# In[103]:


# Rank discussions by upvotes
print(df.sort_values(by = ['votes'], ascending=False)['title'])


# ## Experiment scraping discussion text

# In[121]:


# Function 
def google(post_id):
    # URL to scrape from taking in user input
    url = "https://www.kaggle.com/c/ashrae-energy-prediction/discussion/" + str(post_id)
    dcap = dict(DesiredCapabilities.PHANTOMJS)
    dcap["phantomjs.page.settings.userAgent"] = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.98 Safari/537.36 OPR/44.0.2510.857")
    # Use headless chrome as the webdriver
    driver = webdriver.Chrome(desired_capabilities=dcap, service_args=['--ignore-ssl-errors=true','--ssl-protocol=any'])
    driver.get(url)
    print(url)
    # Wait for page to load
    wait = WebDriverWait(driver, 30)
    wait
    # Save a screenshot of the page being scraped
    #driver.save_screenshot(r'image.png')
    s = BeautifulSoup(driver.page_source, "lxml")
    pretty_s = s.prettify()
    print(pretty_s)
    driver.close()
    # Find the html of interest with the following tag
    best_price_tags = s.findAll('div', 'topic__content')
    best_prices = []
    for tag in best_price_tags:
        best_prices.append(tag.text)
        #print(best_prices)

    flight_info  = OrderedDict() 
    lists=[]
    
    # Create a list including the following details for each flight
    for x in best_prices:

        flight_info={'post':x
                }
        lists.append(flight_info)

    labels = ['post']
    
    # Create a dataframe from the list
    df = pd.DataFrame.from_records(lists, columns=labels)
    
    return df


# In[ ]:


# For every discussion 'id', scrape page and extract post content
for post_id in df['id']:
    post_df = google(post_id)
    print(df.index[df['id'] == post_id])
    df.at[df.index[df['id'] == post_id], 'postContent'] = post_df.at[0,'post']

# Write the scraped data to the csv database
df.to_csv('discussions.csv', sep=',', header=True, index=False)

