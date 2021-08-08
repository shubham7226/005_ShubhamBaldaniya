import nltk
from nltk.corpus import twitter_samples
import matplotlib.pyplot as plt
import random

nltk.download('twitter_samples')

all_positive=twitter_samples.strings('positive_tweets.json')
all_negative=twitter_samples.strings('negative_tweets.json')

print('Number of positive:',len(all_positive))
print('Number of negative ',len(all_negative))

print('\nthe type of all positive : ',type(all_positive))
print('the type of a tweet entry is: ',type(all_negative[0]))

fig=plt.figure(figsize=(5,5))

labels='ML-BSB-Lec','ML-HAP-Lec','ML-HAP-Lab'

sizes=[40,35,10]

plt.pie(sizes,labels=labels,autopct='%.2f%%',shadow=True,startangle=90)

plt.axis('equal')
plt.show()

fig=plt.figure(figsize=(8,5))

labels='Positives','Negatives'

sizes=[len(all_positive),len(all_negative)]

plt.pie(sizes,labels=labels,autopct='%1.1f%%',shadow=True,startangle=90)

plt.axis('equal')
plt.show()

print('\033[92m'+all_positive[random.randint(0,5000)])

print('\033[91m'+all_negative[random.randint(0,5000)])

tweet=all_positive[227]

nltk.download('stopwords')



import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

print('\033[92m'+tweet)

tweet2=re.sub(r'https?:\/\/.*[\r\n]','',tweet)

tweet2=re.sub(r'#','',tweet2)

print(tweet2)

print()
print('\033[92m'+tweet2)
print('\033[94m')

tokenizer=TweetTokenizer(preserve_case=False)

tweet_tokens=tokenizer.tokenize(tweet2)

print()
print('Tokenized string:')
print(tweet_tokens)

stopwords_english=stopwords.words('english')

print('Stop words\n')
print(stopwords_english)



print()

print('\033[92m')
print(tweet_tokens)
print('\033[94m')

tweets_clean=[]

for word in tweet_tokens:
    if (word not in stopwords_english and word not in string.punctuation):
        tweets_clean.append(word)
    
print('removed stop words and punctuation:')

print(tweets_clean)

print()
print('\033[92m')
print(tweets_clean)
print('\033[94m')

stemmer=PorterStemmer()

tweets_stem=[]

for word in tweets_clean:
    stem_word=stemmer.stem(word)
    tweets_stem.append(stem_word)

print('stemmed words:')
print(tweets_stem)