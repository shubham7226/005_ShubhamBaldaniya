#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Replace Manual version of Logistic Regression with TF based version.


# In[1]:


import nltk
from nltk.corpus import twitter_samples 
import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


# In[2]:


nltk.download('twitter_samples')
nltk.download('stopwords')


# In[3]:


import re
import string
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer


# In[4]:


all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')


# In[5]:


test_pos = all_positive_tweets[4000:]
train_pos = all_positive_tweets[:4000]
test_neg = all_negative_tweets[4000:]
train_neg = all_negative_tweets[:4000]
train_x = train_pos+train_neg
test_x = test_pos+test_neg


# In[6]:


def process_tweet(tweet):
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')

    tweet = re.sub(r'\$\w*', '', tweet)
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    tweet = re.sub(r'#', '', tweet)
   
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
            if(word not in stopwords_english and word not in string.punctuation):
              steam_word=stemmer.stem(word)
              tweets_clean.append(steam_word)
    return tweets_clean


# In[7]:


def build_freqs(tweets, ys):
  yslist = np.squeeze(ys).tolist()
  freqs = {}
  for y, tweet in zip(yslist, tweets):
    for word in process_tweet(tweet):
      pair = (word, y)
      if pair in freqs:
        freqs[pair] += 1
      else:
        freqs[pair] = 1
  return freqs


# In[8]:


train_y = np.append(np.ones((len(train_pos), 1)), np.zeros((len(train_neg), 1)), axis=0)
test_y = np.append(np.ones((len(test_pos), 1)), np.zeros((len(test_neg), 1)), axis=0)


# In[9]:


freqs = build_freqs(train_x,train_y)

print("\ntype(freqs) = " + str(type(freqs)))
print("len(freqs) = " + str(len(freqs.keys())))


# In[10]:


print('Example of positive tweet: \n', train_x[0])
print('\nExample of the processed version of the tweet: \n', process_tweet(train_x[0]))


# In[11]:


def extract_features(tweet, freqs):
 
    word_l = process_tweet(tweet)
    x = np.zeros((1, 2)) 
    
    for word in word_l:
        if((word,1) in freqs):
          x[0,0]+=freqs[word,1]
    
        if((word,0) in freqs):
          x[0,1]+=freqs[word,0]
    
    assert(x.shape == (1, 2))
    return x[0]


# In[12]:


# test 1 : 0n training data 
tmp1 = extract_features(train_x[0], freqs)
print(tmp1)


# In[13]:


# test 2: check for when the words are not in the freqs dictionary
tmp2 = extract_features('happy', freqs)
print(tmp2)


# In[14]:


def predict_tweet(tweet):
  with tf.Session() as sess:
      saver.restore(sess,save_path='TSession')
      data_i=[]
      for t in tweet:
        data_i.append(extract_features(t,freqs))
      data_i=np.asarray(data_i)
      return sess.run(tf.nn.sigmoid(tf.add(tf.matmul(a=data_i,b=W,transpose_b=True),b)))
      print("Fail")
  return 


# In[15]:


b=tf.Variable(np.random.randn(1),name="Bias")
W=tf.Variable(np.random.randn(1,2),name="Bias")


# In[16]:


data=[]
for t in train_x:
  data.append(extract_features(t,freqs))
data=np.asarray(data)


# In[17]:


Y_hat = tf.nn.sigmoid(tf.add(tf.matmul(np.asarray(data), W,transpose_b=True), b)) 
print(Y_hat)
ta=np.asarray(train_y)
cost = tf.nn.sigmoid_cross_entropy_with_logits( 
                    logits = Y_hat, labels = ta) 
print("\n",cost)


# In[18]:


optimizer = tf.train.GradientDescentOptimizer(learning_rate = 1e-4,name="GradientDescent").minimize(cost) 
init = tf.global_variables_initializer() 


# In[19]:


saver = tf.train.Saver()
with tf.Session() as sess:
  
  sess.run(init)
  print("Bias",sess.run(b))
  print("Weight",sess.run(W))
  for epoch in range(400):
    sess.run(optimizer)
    preds=sess.run(Y_hat)
    acc=((preds==ta).sum())/len(train_y)
    accu=[]
    repoch=False
    if repoch:
      accu.append(acc)
    if epoch % 1000 == 0:
      print("Accuracy",acc)
    saved_path = saver.save(sess, 'TSession')


# In[20]:


preds=predict_tweet(test_x)
print(preds,len(test_y))


# In[21]:


def calculate_accuracy(x,y):
  if len(x)!=len(y):
    print("dimensions are different")
    return
  return ((x==y).sum())/len(y)


# In[22]:


print(calculate_accuracy(preds,test_y))

