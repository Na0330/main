#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


# In[2]:


import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')


# In[3]:


import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer


# In[10]:


# 读取 csv文件
xhs_df = pd.read_csv('D:\\布里斯托大学\\TB2\\EFIMM0139 - Social Media and Web Analytics\\Assessment\\data\\time.csv')
# 将数据保存为JSON文件
xhs_df.to_json('data.json', orient='records')
print(xhs_df.head())


# In[5]:


# Read the JSON file
df = pd.read_json('data.json')
print(df.head())


# In[11]:


df = df.drop(columns=['Unnamed: 15'])


# In[20]:


df.dtypes


# In[13]:


df['desc'] = df['desc'].astype(str)


# In[22]:


# Calculate the top 5 ip location
location_df = df.groupby(by=['note_ip_location']).size().reset_index(name='count').sort_values(by='count', ascending=False).head(5)
location_df


# In[23]:


location_df['note_ip_location'] = location_df['note_ip_location'].str.replace("\n",'')
location_df


# In[25]:


# Install Chinese fonts
import matplotlib.pyplot as plt

# Set Chinese font to SimHei (Song typeface)
plt.rcParams['font.sans-serif'] = ['SimHei']  # Specify default font
plt.rcParams['axes.unicode_minus'] = False  # Fixed an issue where the save image is displayed as a square with the negative sign '-'


# In[29]:


import numpy as np
import matplotlib.pyplot as plt

location_df.plot(kind='bar', x='note_ip_location', y='count',rot=0)
plt.xlabel('ip_location')  
plt.ylabel('Count')     
plt.title('Distribution of ip_location (Top 5)')  

# Add numbers in bar chart
for index, value in enumerate(location_df['count']):
    plt.text(index, value, str(value), ha='center', va='bottom')
    
# Legend is not displayed
plt.legend().remove()
plt.show()  


# In[30]:


# Find popular users and content
popularity_df = df[['user_id','title','desc','liked_count', 'collected_count', 'comment_count', 'share_count']]
popularity_df.head()


# In[35]:


# Sort by like_count
popularity_df = popularity_df.sort_values(by=['liked_count'],ascending=False)
# Display desc complete data
pd.set_option('display.max_colwidth', None)
popularity_df.head()


# tag list analysis

# In[60]:


tag_texts = df['tag_list'].tolist()
print(tag_texts)


# In[67]:


# 创建一个空字典来存储每个带引号中文本的出现次数
quoted_text_counts = {}

# 遍历 DataFrame 中的每个 tag_list 值
for tags in df['tag_list']:
    # 使用正则表达式匹配带引号中的文本
    quoted_texts = re.findall(r"'(.*?)'", tags)
    # 遍历匹配到的带引号中文本
    for text in quoted_texts:
        # 统计每个带引号中文本出现的次数
        quoted_text_counts[text] = quoted_text_counts.get(text, 0) + 1

# 输出结果
for text, count in quoted_text_counts.items():
    print(f"'{text}' ： {count} ")


# In[72]:


import matplotlib.pyplot as plt

# Descending order of occurrence times
sorted_counts = sorted(quoted_text_counts.items(), key=lambda x: x[1], reverse=True)

# Select the top 20 
top_20 = sorted_counts[:20]

# Extract the label and the number of occurrences
labels = [item[0] for item in top_20]
counts = [item[1] for item in top_20]

# Draw bar charts
plt.figure(figsize=(10, 6))
bars = plt.bar(labels, counts,color='skyblue')
plt.xlabel('Tag list')
plt.ylabel('Counts')
plt.title('Top 20 occurrences in the tag list')
plt.xticks(rotation=45, ha='right')

# Add a label above each bar
for bar, count in zip(bars, counts):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(count), ha='center', va='bottom')

plt.tight_layout()
plt.show()


# In[ ]:




