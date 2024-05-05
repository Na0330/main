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
# 将NaN值替换为默认值（例如中性）
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


negative_ip_location = negative_rows.groupby(by=['note_ip_location']).size().reset_index(name='count').sort_values(by='count', ascending=False).head(5)
negative_ip_location


# In[9]:


negative_ip_location['note_ip_location'] = negative_ip_location['note_ip_location'].str.replace("\n",'')
negative_ip_location


# In[10]:


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

# 创建 SnowballStemmer 对象
stemmer = SnowballStemmer('english')

# 分词并进行词干提取的函数
def stemming_on_text(text):
    # 分词
    seg_list = jieba.cut(text, cut_all=False)
    # 词干提取
    stemmed_text = " ".join([stemmer.stem(word) for word in seg_list])
    return stemmed_text

# 对 'text' 列中的每个句子应用分词和词干提取函数
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
    # 使用正则表达式替换要去除的单词
    for word in words_to_remove:
        text = re.sub(r'\b' + re.escape(word) + r'\b', '', text)
    return text

# 要去除的单词列表
words_to_remove = ['的', '了','是','R','我','穿','优衣库','很','都','也','买','有','就','真','还',
                      '在','没','但','们','点','这个','这','去','和','不','一','款','哭','感觉','上',
                       '啊','一下','要','太','件','个','试','下','能','条','又','更','说','给','吧','大'
                       ,'身','会','来','谁','过','出','你','啦','只','人','到','惹','也','我',']','r','🤔','❤'
                ,'⭐','🆘','❓','…','✅','‼','🤩','✨','🫡','❤️','🔎','如','图','题''哦','呀','吗'
                ,'才','日','做','再']

# 在 DataFrame 的 text 列上应用函数
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


pyLDAvis.save_json(vis, 'data.json')


# In[38]:


import json
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


# In[39]:


# 假设pyLDAvis输出的data.json文件在当前目录
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


# 应用预处理，并分词、词干提取
from nltk.stem import PorterStemmer
nltk.download('punkt', quiet=True)
ps = PorterStemmer()


# In[45]:


from collections import Counter

# 创建一个空的计数器
word_counts = Counter()

# 处理每个处理后的分词结果
for processed_text in desc_texts:
    # 将每个处理后的分词结果拆分成单词列表
    words = processed_text.split()
    # 更新计数器
    word_counts.update(words)

# 输出每个单词的出现频率
for word, count in word_counts.items():
    print(f"'{word}' , {count}")


# In[46]:


# 计算总词数
total_word_count = sum(word_counts.values())
print("Total word count:", total_word_count)


# In[47]:


# 计算文档频率
document_frequency = Counter()

# 处理每个处理后的分词结果
for processed_text in desc_texts:
    # 将每个处理后的分词结果转换为set，以去除重复词语
    unique_words = set(processed_text.split())
    # 更新文档频率计数器
    document_frequency.update(unique_words)

# 输出每个单词的文档频率
for word, freq in document_frequency.items():
    print(f"'{word}' , {freq}")


# In[48]:


import math

# 计算每个单词的TF-IDF值
def calculate_tfidf(word_counts, total_word_count, document_frequency):
    tfidf_scores = {}
    num_documents = len(word_counts)

    for word, count in word_counts.items():
        # 计算TF（词频）
        tf = count / total_word_count

        # 计算IDF（逆文档频率）
        idf = math.log(num_documents / document_frequency[word])

        # 计算TF-IDF
        tfidf_scores[word] = tf * idf

    return tfidf_scores

# 假设document_frequency是一个字典，包含每个单词的文档频率
# document_frequency[word]表示包含单词word的文档数
tfidf_scores = calculate_tfidf(word_counts, total_word_count, document_frequency)
tfidf_scores


# In[128]:


# 获取最大的TF-IDF值
max_tfidf = max(tfidf_scores.values())

# 归一化TF-IDF值
normalized_tfidf_scores = {word: tfidf_score / max_tfidf for word, tfidf_score in tfidf_scores.items()}
normalized_tfidf_scores


# In[64]:


import networkx as nx
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 根据阈值构建语义网络化图,限制节点数量为100
def build_semantic_network(normalized_tfidf_scores, threshold=0.5, max_nodes=100):
    G = nx.Graph()

    # 获取前100个单词及其TF-IDF得分
    top_words = sorted(normalized_tfidf_scores.items(), key=lambda x: x[1], reverse=True)[:max_nodes]

    # 添加节点
    for word, score in top_words:
        G.add_node(word)
        
    # 添加边
    for i, (word1, score1) in enumerate(top_words):
        for j, (word2, score2) in enumerate(top_words):
            if i != j:
                similarity = calculate_similarity(score1, score2)
                if similarity >= threshold:
                    G.add_edge(word1, word2, weight=similarity)

    return G

# 计算单词之间的相似度
def calculate_similarity(score1, score2):
    # 假设相似度等于两个得分的差的绝对值
    similarity = abs(score1 - score2)
    return similarity

# 使用 TF-IDF 得分构建语义网络化图，限制节点数量为100
semantic_network = build_semantic_network(normalized_tfidf_scores, max_nodes=100)

# 绘制网络图
pos = nx.spring_layout(semantic_network, k=1)# k参数控制节点之间的弹簧力度，增加此值会增加节点之间的间距
    
# 绘制网络图
plt.figure(figsize=(10, 8), facecolor='white')
nx.draw(semantic_network, pos, node_color='skyblue', node_size=700, 
        edge_color='gray', with_labels=True)
plt.box(False)
plt.axis('off')
plt.show()


# In[67]:


import networkx as nx
import matplotlib.pyplot as plt

# 假设您已经有了每个单词的出现频率，存储在字典 frequency_dict 中，其中键是单词，值是频率
# 示例：
frequency_dict = word_counts

# 根据阈值构建语义网络化图，限制节点数量为100
def build_semantic_network(normalized_tfidf_scores, frequency_dict, threshold=0.5, max_nodes=100):
    G = nx.Graph()

    # 获取前100个单词及其TF-IDF得分
    top_words = sorted(normalized_tfidf_scores.items(), key=lambda x: x[1], reverse=True)[:max_nodes]

    # 添加节点，并为每个节点添加频率属性
    for word, score in top_words:
        G.add_node(word, frequency=frequency_dict.get(word, 0))  # 使用get方法获取频率，如果没有找到，默认为0
        
    # 添加边
    for i, (word1, score1) in enumerate(top_words):
        for j, (word2, score2) in enumerate(top_words):
            if i != j:
                similarity = calculate_similarity(score1, score2)
                if similarity >= threshold:
                    G.add_edge(word1, word2, weight=similarity)

    return G

# 计算单词之间的相似度
def calculate_similarity(score1, score2):
    # 假设相似度等于两个得分的差的绝对值
    similarity = abs(score1 - score2)
    return similarity

# 使用 TF-IDF 得分构建语义网络化图，限制节点数量为100
semantic_network = build_semantic_network(normalized_tfidf_scores, frequency_dict, max_nodes=100)

# 绘制网络图
pos = nx.spring_layout(semantic_network, k=1.2)

# 获取节点的频率属性，作为节点大小的依据
node_sizes = [data['frequency'] for node, data in semantic_network.nodes(data=True)]

# 绘制网络图，设置节点大小为交互频率的大小
plt.figure(figsize=(10, 8), facecolor='white')
nx.draw(semantic_network, pos, node_color='skyblue', node_size=node_sizes, 
        edge_color='gray', with_labels=True)
plt.box(False)
plt.axis('off')
plt.show()



# In[ ]:




