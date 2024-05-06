#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install WordCloud')
import re
import numpy as np
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# In[2]:


# Importing the dataset

df = pd.read_csv('D:\\布里斯托大学\\TB2\\EFIMM0139 - Social Media and Web Analytics\\Assessment\\data\\sentiment_time.csv')


# In[3]:


df.head()


# In[4]:


lab_to_sentiment = {0.0:"Negative", 1.0:"Neutral", 2.0:"Positive"}
def label_decoder(label):
  return lab_to_sentiment[label]
# Replace the NaN value with the default value (such as neutral)
df['sentiment'] = df['sentiment'].fillna(1.0)
df.sentiment = df.sentiment.apply(lambda x: label_decoder(x))
df.head()


# In[5]:


df.sentiment.value_counts()


# In[6]:


#Calculate the frequency distribution of emotional values
val_count = df.sentiment.value_counts()

# Draw a bar chart
plt.figure(figsize=(8,4))
plt.bar(val_count.index, val_count.values)

plt.title("Sentiment Data Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Counts")


# In[7]:


negative_rows = df[df['sentiment'] == 'Negative']
negative_rows


# In[8]:

# Select the top 5 ip locations and count them
negative_ip_location = negative_rows.groupby(by=['note_ip_location']).size().reset_index(name='count').sort_values(by='count', ascending=False).head(5)
negative_ip_location


# In[9]:


negative_ip_location['note_ip_location'] = negative_ip_location['note_ip_location'].str.replace("\n",'')
negative_ip_location


# In[10]:

# Combine title and desc into a text column
df['text'] = df['title'] + ' ' + df['desc']
df


# In[11]:


import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer


# In[12]:


# Remove punctuations
def remove_punctuations(data):
    return re.sub(r"[~.,，%/:;|?_&+*=!！[：×？#‘’；（）` • 。...-~—]"," ",data)
df['text']= df['text'].apply(lambda x: remove_punctuations(x))


# In[13]:


import jieba
from nltk.stem import SnowballStemmer

# Create the SnowballStemmer object
stemmer = SnowballStemmer('english')

# Word segmentation and stem extraction
def stemming_on_text(text):
    # Word segmentation
    seg_list = jieba.cut(text, cut_all=False)
    # Stem extracting
    stemmed_text = " ".join([stemmer.stem(word) for word in seg_list])
    return stemmed_text

# Apply word segmentation and stem extraction functions to each sentence in the 'text' column
df['text'] = df['text'].apply(stemming_on_text)


# In[15]:


# Cleaning and removing the above stop words list from the little red book text
stop_words = stopwords.words('english')
new_stopwords = ['的', '了','是','R','我','穿','优衣库','很','都','也','买','有','就','真','还',
                      '在','没','但','们','点','这个','这','去','和','不','一','款','哭','感觉','上',
                       '啊','一下','要','太','件','个','试','下','能','条','又','更','说','给','吧','大'
                 ,'身','会','来','谁','过','出','你','啦','只','人','到','惹',']','r','🤔','❤'
                 ,'⭐','🆘','❓','…','✅','‼','🤩','✨','🫡','🔎','如','图','题','哦','呀','吗'
                ,'才','日','做','再']
stop_words.extend(new_stopwords)
# Define function to remove Chinese emojis
def remove_chinese_emojis(text):
    emoji_pattern = re.compile("["
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)
def cleaning_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in stop_words])
df['text'] = df['text'].apply(lambda text: cleaning_stopwords(text))
df['text'].head()


# In[16]:


# Cleaning and removing the nonsense words from the little red book text
import re

def remove_words(text, words_to_remove):
    # Use regular expressions to replace the words you want to remove
    for word in words_to_remove:
        text = re.sub(r'\b' + re.escape(word) + r'\b', '', text)
    return text

# List of words to remove
words_to_remove = ['的', '了','是','R','我','穿','优衣库','很','都','也','买','有','就','真','还',
                      '在','没','但','们','点','这个','这','去','和','不','一','款','哭','感觉','上',
                       '啊','一下','要','太','件','个','试','下','能','条','又','更','说','给','吧','大'
                       ,'身','会','来','谁','过','出','你','啦','只','人','到','惹','也','我',']','r','🤔','❤'
                ,'⭐','🆘','❓','…','✅','‼','🤩','✨','🫡','❤️','🔎','如','图','题''哦','呀','吗'
                ,'才','日','做','再']

# Apply the function to the text column of the DataFrame
df['text'] = df['text'].apply(lambda x: remove_words(x, words_to_remove))
df['text'].head()


# In[17]:


# Remove numbers
def remove_numbers(data):
    return re.sub('[0-9]+', '', data)
df['text'] = df['text'].apply(lambda x: remove_numbers(x))


# In[18]:


df.head()


# In[19]:


new_df = df[['sentiment', 'text']].copy()
new_df


# In[20]:


# Positive wordcloud

from wordcloud import WordCloud

# Insert Chinese font
font_path = 'C:/Windows/Fonts/Deng.ttf'
plt.figure(figsize = (20,20)) 
wc = WordCloud(font_path=font_path, background_color = 'white',max_words = 2000 , width = 1600 , height = 800).generate(" ".join(df[df.sentiment == 'Positive'].text))
plt.imshow(wc , interpolation = 'bilinear')


# In[21]:


# Negative wordcloud
font_path = 'C:/Windows/Fonts/Deng.ttf'
plt.figure(figsize = (20,20)) 
wc = WordCloud(font_path=font_path, max_words = 2000 , width = 1600 , height = 800).generate(" ".join(df[df.sentiment == 'Negative'].text))
plt.imshow(wc , interpolation = 'bilinear')


# LDA Analysis

# In[24]:


import pandas as pd
import unicodedata
import re
import contractions
import string
get_ipython().system('pip install Gensim')
import gensim
import gensim.corpora as corpora
get_ipython().system('pip install spacy')
import spacy
get_ipython().system('pip install pyLDAvis')
import pyLDAvis
import pyLDAvis.gensim_models


# In[30]:


df_no_nans=new_df


# In[31]:


# Generating the document-term matrix and dictionary
def generate_tokens(tweet):
    words=[]
    for word in tweet.split(' '):
    # using the if condition because we introduced extra spaces during text cleaning
        if word!='':
           words.append(word)
    return words
#storing the generated tokens in a new column named 'tokens'
#df_no_nans['tokens']=df_no_nans.text.apply(generate_tokens)
df_no_nans.loc[:, 'tokens'] = df_no_nans['text'].apply(generate_tokens)


# In[32]:


df_no_nans['tokens'][0]


# In[33]:


def create_dictionary(words):
    return corpora.Dictionary(words)
id2word=create_dictionary(df_no_nans['tokens'])
print(id2word)


# In[34]:


def create_document_matrix(tokens,id2word):
    corpus = []
    for text in tokens:
        corpus.append(id2word.doc2bow(text))
    return corpus
#passing the dataframe column having tokens and dictionary
corpus=create_document_matrix(df_no_nans['tokens'],id2word)
print(df_no_nans['tokens'][0])
print(corpus[0])


# In[35]:


lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=6,
                                            random_state=100,
                                             )


# In[36]:


pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word, mds="mmds", R=10)
vis


# In[37]:

# Save the LDA data as a JSON format file
pyLDAvis.save_json(vis, 'data.json')


# In[38]:


import json
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


# In[39]:


# Read the data.json file
with open('data.json', 'r') as f:
    data = json.load(f)


# In[40]:


print(data.keys())


# In[41]:


import os
import math
import time
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm
import re

from gensim.models import KeyedVectors
import gensim.downloader as api

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.express as px

# Below libraries are for text processing using NLTK
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Below libraries are for feature representation using sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Below libraries are for similarity matrices using sklearn
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances


# In[42]:


# Gets all the text for the desc column
desc_texts = new_df['text'].tolist()
print(desc_texts)


# In[43]:


# Output all text
for text in desc_texts:
    print(text)


# In[44]:


# Applying preprocessing, word segmentation and stem extraction
from nltk.stem import PorterStemmer
nltk.download('punkt', quiet=True)
ps = PorterStemmer()


# In[45]:

# Count the frequency of each word
from collections import Counter

# Create an empty counter
word_counts = Counter()

# Process each processed participle
for processed_text in desc_texts:
    # Split each processed word segmentation result into word lists
    words = processed_text.split()
    # Refresh counter
    word_counts.update(words)

# Output the frequency of each word
for word, count in word_counts.items():
    print(f"'{word}' , {count}")


# In[46]:


# Count the total number of words
total_word_count = sum(word_counts.values())
print("Total word count:", total_word_count)


# In[47]:


# Calculate document frequency
document_frequency = Counter()

# Process each processed participle
for processed_text in desc_texts:
    # Each processed word segmentation is converted to a set to remove duplicate words
    unique_words = set(processed_text.split())
    # Update the document frequency counter
    document_frequency.update(unique_words)

# Output the document frequency for each word
for word, freq in document_frequency.items():
    print(f"'{word}' , {freq}")


# In[48]:


import math

# Calculate the TF-IDF value for each word
def calculate_tfidf(word_counts, total_word_count, document_frequency):
    tfidf_scores = {}
    num_documents = len(word_counts)

    for word, count in word_counts.items():
        # Calculate TF
        tf = count / total_word_count

        # Calculate IDF
        idf = math.log(num_documents / document_frequency[word])

        # calculate TF-IDF
        tfidf_scores[word] = tf * idf

    return tfidf_scores


# Output TF-IDF
tfidf_scores = calculate_tfidf(word_counts, total_word_count, document_frequency)
tfidf_scores


# In[128]:


# Gets the maximum TF-IDF value
max_tfidf = max(tfidf_scores.values())

# Normalized TF-IDF values
normalized_tfidf_scores = {word: tfidf_score / max_tfidf for word, tfidf_score in tfidf_scores.items()}
normalized_tfidf_scores


# In[64]:


import networkx as nx
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # Set the Chinese font to bold
plt.rcParams['axes.unicode_minus'] = False  # Solve the negative sign display problem

# The semantic network graph is constructed according to the threshold value, and the number of restricted nodes is 100
def build_semantic_network(normalized_tfidf_scores, threshold=0.5, max_nodes=100):
    G = nx.Graph()

    # Get the top 100 words and their TF-IDF scores
    top_words = sorted(normalized_tfidf_scores.items(), key=lambda x: x[1], reverse=True)[:max_nodes]

    # Add node
    for word, score in top_words:
        G.add_node(word)
        
    # Add edge
    for i, (word1, score1) in enumerate(top_words):
        for j, (word2, score2) in enumerate(top_words):
            if i != j:
                similarity = calculate_similarity(score1, score2)
                if similarity >= threshold:
                    G.add_edge(word1, word2, weight=similarity)

    return G

# Calculate the similarity between words
def calculate_similarity(score1, score2):
    # Suppose the similarity is equal to the absolute value of the difference between the two scores
    similarity = abs(score1 - score2)
    return similarity

# The TF-IDF score was used to construct a semantic network graph with a limit of 100 nodes
semantic_network = build_semantic_network(normalized_tfidf_scores, max_nodes=100)

# Draw a network map
pos = nx.spring_layout(semantic_network, k=1) # The k parameter controls the force of the spring between the nodes, increasing this value increases the spacing between the nodes
    
plt.figure(figsize=(10, 8), facecolor='white')
nx.draw(semantic_network, pos, node_color='skyblue', node_size=700, 
        edge_color='gray', with_labels=True)
plt.box(False)
plt.axis('off')
plt.show()


# In[67]:


import networkx as nx
import matplotlib.pyplot as plt

# Store the frequency of each word in the dictionary frequency_dict
frequency_dict = word_counts

# The semantic network graph is constructed according to the threshold value, and the number of restricted nodes is 100
def build_semantic_network(normalized_tfidf_scores, frequency_dict, threshold=0.5, max_nodes=100):
    G = nx.Graph()

    # Get the top 100 words and their TF-IDF scores
    top_words = sorted(normalized_tfidf_scores.items(), key=lambda x: x[1], reverse=True)[:max_nodes]

    # Add node
    for word, score in top_words:
        G.add_node(word, frequency=frequency_dict.get(word, 0))  # Use the get method to get the frequency, if not found, the default is 0

        
    # Add edge
    for i, (word1, score1) in enumerate(top_words):
        for j, (word2, score2) in enumerate(top_words):
            if i != j:
                similarity = calculate_similarity(score1, score2)
                if similarity >= threshold:
                    G.add_edge(word1, word2, weight=similarity)

    return G

# Calculate the similarity between words
def calculate_similarity(score1, score2):
    # Suppose the similarity is equal to the absolute value of the difference between the two scores
    similarity = abs(score1 - score2)
    return similarity

# The TF-IDF score was used to construct a semantic network graph with a limit of 100 nodes
semantic_network = build_semantic_network(normalized_tfidf_scores, frequency_dict, max_nodes=100)

# Draw a network map
pos = nx.spring_layout(semantic_network, k=1.2)

# Gets the frequency attribute of the node as a basis for the size of the node
node_sizes = [data['frequency'] for node, data in semantic_network.nodes(data=True)]

# Draw a network map and set the node size to the interaction frequency
plt.figure(figsize=(10, 8), facecolor='white')
nx.draw(semantic_network, pos, node_color='skyblue', node_size=node_sizes, 
        edge_color='gray', with_labels=True)
plt.box(False)
plt.axis('off')
plt.show()



# In[ ]:




