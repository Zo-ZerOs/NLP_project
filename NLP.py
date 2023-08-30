import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer   
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import nltk
from nltk.corpus import stopwords
import string
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.tag import StanfordNERTagger
from itertools import chain
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer
df = pd.read_csv('File_Name')
df2 = pd.DataFrame.copy(df,deep = True)
first = df2.drop(['Blinding of intervention','Classes'], axis=1)
for i in range(628):
    text1 = first['Blinding of Outcome assessment'][i]
    text2 = first['text'][i]
#words1 = text1.lower()                      # change to lowercase
    words2 = text2.lower()
#print(words2)
#re1 = words1.split()                        # change to array
    re2 = words2.split()
#print(re2)
    table = str.maketrans("","",string.punctuation) #remove punctuation
#stripped1 = [w.translate(table) for w in re1]
    stripped2 = [w.translate(table) for w in re2]
#print(stripped1)
#print(stripped2)
#result1 = [item for item in stripped1 if item.isalpha()]    #remove number
    result2 = [item for item in stripped2 if item.isalpha()]
    from nltk.stem import WordNetLemmatizer                 # lemma
    
    from nltk.corpus import stopwords    
    stop_words = set(stopwords.words('english'))                            # stop word 
#word_tokens = word_tokenize(example_sent) # word_tokenize is result2  
    filtered_sentence = [w for w in result2 if not w in stop_words]   
    filtered_sentence = []  
    for w in result2: 
        if w not in stop_words: 
            filtered_sentence.append(w)                                 # remove some word
    test = [string for string in filtered_sentence if string !='a' and string != 'b' and string != 'c' and string!= 'd' and string!= 'e' and string!= 'f' and string!= 'g' and string!= 'h' ]
    test = [string for string in test if string !='i'and string!= 'j' and string!= 'k' and string!= 'l' and string!= 'm' and string!= 'n' and string!= 'o' and string!= 'p' ]
    test = [string for string in test if string !='q'and string!= 'r' and string!= 's' and string!= 't' and string!= 'u' and string!= 'v' and string!= 'w' and string!= 'x' and string!='y' and string!='z' ]
    test = [string for string in test if string !='al' and string!= 'et' and string!= 'kf' and string!= 'td' and string!= 'gf' and string!= 'ef' and string!='vol' and string!='usa' and string!='us' and string!='uk' and string!='ann']    
#print(test)
    lemmatizer = WordNetLemmatizer()                                # lemmatizer
# function to convert nltk tag to wordnet tag
    def nltk_tag_to_wordnet_tag(nltk_tag):
        if nltk_tag.startswith('J'):
            return wordnet.ADJ
        elif nltk_tag.startswith('V'):
            return wordnet.VERB
        elif nltk_tag.startswith('N'):
            return wordnet.NOUN
        elif nltk_tag.startswith('R'):
            return wordnet.ADV
        else:          
            return None

    def lemmatize_sentence(sentence):
        #tokenize the sentence and find the POS tag for each token
        nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))  
    #tuple of (token, wordnet_tag)
        wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
        lemmatized_sentence = []
        for word, tag in wordnet_tagged:
            if tag is None:
            #if there is no available tag, append the token as is
                lemmatized_sentence.append(word)
            else:        
            #else use the tag to lemmatize the token
                lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
        return " ".join(lemmatized_sentence)
    x = " ".join(test)                                              # change array to text 
    first['text'][i] = lemmatize_sentence(x)
token = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv = CountVectorizer(ngram_range = (1,1),tokenizer = token.tokenize)
text_counts= cv.fit_transform(first['text'])
from sklearn.model_selection import train_test_split
X_train = text_counts
y_train = first['Blinding of Outcome assessment']
#X_train, X_test, y_train, y_test = train_test_split(
#    text_counts, first['Blinding of intervention'], test_size=0.2, random_state=1)
clf = MultinomialNB().fit(X_train, y_train)

import pandas as pd
df11 = pd.read_csv('File_Name')
for i in range(397):
    text2 = df11['text'][i]
#words1 = text1.lower()                      # change to lowercase
    words2 = text2.lower()
#print(words2)
#re1 = words1.split()                        # change to array
    re2 = words2.split()
#print(re2)
    table = str.maketrans("","",string.punctuation) #remove punctuation
#stripped1 = [w.translate(table) for w in re1]
    stripped2 = [w.translate(table) for w in re2]
#print(stripped1)
#print(stripped2)
#result1 = [item for item in stripped1 if item.isalpha()]    #remove number
    result2 = [item for item in stripped2 if item.isalpha()]
    from nltk.stem import WordNetLemmatizer                 # lemma
    
    from nltk.corpus import stopwords    
    stop_words = set(stopwords.words('english'))                            # stop word 
#word_tokens = word_tokenize(example_sent) # word_tokenize is result2  
    filtered_sentence = [w for w in result2 if not w in stop_words]   
    filtered_sentence = []  
    for w in result2: 
        if w not in stop_words: 
            filtered_sentence.append(w)                                 # remove some word
    test = [string for string in filtered_sentence if string !='a' and string != 'b' and string != 'c' and string!= 'd' and string!= 'e' and string!= 'f' and string!= 'g' and string!= 'h' ]
    test = [string for string in test if string !='i'and string!= 'j' and string!= 'k' and string!= 'l' and string!= 'm' and string!= 'n' and string!= 'o' and string!= 'p' ]
    test = [string for string in test if string !='q'and string!= 'r' and string!= 's' and string!= 't' and string!= 'u' and string!= 'v' and string!= 'w' and string!= 'x' and string!='y' and string!='z' ]
    test = [string for string in test if string !='al' and string!= 'et' and string!= 'kf' and string!= 'td' and string!= 'gf' and string!= 'ef' and string!='vol' and string!='usa' and string!='us' and string!='uk' and string!='ann']    
#print(test)
    lemmatizer = WordNetLemmatizer()                                # lemmatizer
# function to convert nltk tag to wordnet tag
    x = " ".join(test)                                              # change array to text 
    df11['text'][i] = lemmatize_sentence(x)
token2 = RegexpTokenizer(r'[a-zA-Z0-9]+')
cv2 = CountVectorizer(ngram_range = (1,1),tokenizer = token2.tokenize)
text_counts2= cv.transform(df11['text'])
predicted = clf.predict(text_counts2)
print(predicted)
#print("MultinomialNB Accuracy:",metrics.accuracy_score(y_test, predicted))

