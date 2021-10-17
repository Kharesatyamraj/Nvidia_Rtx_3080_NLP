# -*- coding: utf-8 -*-
"""


@author: khare
"""


## Name- SATYAM RAJ KHARE


import requests
#pip install bs4
import pandas as pd
from bs4 import BeautifulSoup as bs 
import re
import matplotlib.pyplot as plt
#conda install -c conda-forge wordcloud
from wordcloud import WordCloud
#pip install nltk
import nltk

#Task 1:
#1.	Extract reviews of any product from e-commerce website Amazon.
#2.	Perform sentiment analysis on this extracted data and build a unigram and bigram word cloud. 



# creating empty reviews list 
Nvidia_reviews=[]


for i in range(1,100):
  Np=[]  
  url="https://www.amazon.in/product-reviews/B079JSKCW3/ref=cm_cr_arp_d_viewopt_sr?ie=UTF8&filterByStar=all_stars&reviewerType=avp_only_reviews&pageNumber=1#reviews-filter-bar"+str(i)
  response = requests.get(url)
  soup = bs(response.content,"html.parser")# creating soup object to iterate over the extracted content 
  reviews = soup.find_all("span",attrs={"class","a-size-base review-text review-text-content"})# Extracting the content under specific tags  
  
  for i in range(len(reviews)):
    Np.append(reviews[i].text)  
 
  Nvidia_reviews=Nvidia_reviews+Np  # adding the reviews of one page to empty list which in future contains all the reviews

# writng reviews in a text file 
with open("Rtx1050Ti.txt","w",encoding='utf8') as text:
    text.write(str(Nvidia_reviews))
import os   
os.getcwd()# get working directory of spyder

# Joinining all the reviews into single paragraph 
Nvidia= " ".join(Nvidia_reviews)



# Removing unwanted symbols incase if exists
Nvidia = re.sub("[^A-Za-z" "]+"," ", Nvidia).lower()
Nvidia = re.sub("[0-9" "]+"," ", Nvidia)

ip_reviews_words = Nvidia.split(" ")



with open("\stop.txt","r") as sw:
    stop_words = sw.read()
    
stop_words = stop_words.split("\n")

#add more words to stop_words
stop_words.extend(['assassin', 'creed','cry','resident','evil','black','witcher','wild','hunt','cod','ghost','fifa'])
ip_reviews_words = [w for w in ip_reviews_words if not w in stop_words]

#ip_reviews_words=pd.DataFrame(pd.Series(ip_reviews_words))
#y =ip_reviews_words.drop_duplicates()
 

# Joinining all the reviews into single paragraph 
Nvidia = " ".join(ip_reviews_words)

# WordCloud can be performed on the string inputs.
# Corpus level word cloud

wordcloud_ip = WordCloud(
                      background_color='lightyellow',
                      width=1800,
                      height=1400
                     ).generate(Nvidia)
plt.title("Unigram WordCloud of Nvidia 1050Ti Reviews  ")
plt.figure(1)
plt.imshow(wordcloud_ip)

# positive words # Choose the path for +ve words stored in system
with open("\positive-words.txt","r") as pos:
  poswords = pos.read().split("\n")


# Positive word cloud
# Choosing the only words which are present in positive words
ip_pos_in_pos = " ".join ([w for w in ip_reviews_words if w in poswords])

wordcloud_pos_in_pos = WordCloud(
                      background_color='White',
                      width=1800,
                      height=1400
                     ).generate(ip_pos_in_pos)
plt.figure(2)
plt.title("Positive Analysis WordCloud of Nvidia 1050Ti Reviews  ");plt.imshow(wordcloud_pos_in_pos)



# negative words Choose path for -ve words stored in system
with open("\positive-words.txt","r") as neg:
  negwords = neg.read().split("\n")
# negative word cloud
# Choosing the only words which are present in negwords
ip_neg_in_neg = " ".join ([w for w in ip_reviews_words if w in negwords])


wordcloud_neg_in_neg = WordCloud(
                      background_color='lightgreen',
                      width=1800,
                      height=1400
                     ).generate(ip_neg_in_neg)

plt.figure(3)
plt.title("Negative Analysis WordCloud of Nvidia 1050Ti Reviews  ");plt.imshow(wordcloud_neg_in_neg)




# wordcloud with bigram
nltk.download('punkt')
from wordcloud import WordCloud, STOPWORDS

WNL = nltk.WordNetLemmatizer()

# Lowercase and tokenize
text = Nvidia.lower()

# Remove single quote early since it causes problems with the tokenizer.
text = text.replace("'", "")

tokens = nltk.word_tokenize(text)
text1 = nltk.Text(tokens)

# Remove extra chars and remove stop words.
text_content = [''.join(re.split("[ .,;:!?‘’``''@#$%^_&*()<>{}~\n\t\\\-]", word)) for word in text1]

# Create a set of stopwords
stopwords_wc = set(STOPWORDS)
customised_words = ['assassin','gta', 'creed','cry','resident','evil','black','witcher','wild','hunt','cod','ghost','fifa','price'] # If you want to remove any particular word form text which does not contribute much in meaning

new_stopwords = stopwords_wc.union(customised_words)#add new words

# Remove stop words
text_content = [word for word in text_content if word not in new_stopwords]

# Take only non-empty entries
text_content = [s for s in text_content if len(s) != 0]

# Best to get the lemmas of each word to reduce the number of similar words
text_content = [WNL.lemmatize(t) for t in text_content]

nltk_tokens = nltk.word_tokenize(text)  
bigrams_list = list(nltk.bigrams(text_content))
print(bigrams_list)

dictionary2 = [' '.join(tup) for tup in bigrams_list]
print (dictionary2)

# Using count vectoriser to view the frequency of bigrams
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(ngram_range=(2, 2))
bag_of_words = vectorizer.fit_transform(dictionary2)
vectorizer.vocabulary_

sum_words = bag_of_words.sum(axis=0)
words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
print(words_freq[:100])

# Generating wordcloud
words_dict = dict(words_freq)
WC_height = 1000
WC_width = 1500
WC_max_words = 200
wordCloud = WordCloud(max_words=WC_max_words, height=WC_height, width=WC_width, stopwords=new_stopwords)
wordCloud.generate_from_frequencies(words_dict)

plt.figure(4)
plt.title('NvidiaReviews \n Most frequently occurring bigrams connected by same colour and font size')
plt.imshow(wordCloud, interpolation='bilinear')
plt.axis("off")
plt.show()

