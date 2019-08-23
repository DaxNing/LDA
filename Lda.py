#!/usr/bin/python
# _*_ coding:UTF-8 _*_

import nltk
from nltk.corpus import stopwords
import io
import string
import  gensim
import re
from gensim import corpora


def doc_complete():
    doc_complete = []
    with io.open("id_article_test.txt","r",encoding='utf-8') as f:
        for line in f:
            doc_complete.append(line.strip())

    return  doc_complete


def data_clean(doc):
   # with io.open(r"cn_stopword.txt","r",encoding='utf-8') as f:
        #stop = f.read()

        stop = set(stopwords.words('indonesian'))
        exclude = set(string.punctuation + '，。、【】“”：；（）《》‘’{}？！⑦()、%^>℃：.”“^-——=&#@￥')
        
        #doc =  doc.lower()
        for i in doc:
            if i in exclude:
                doc = doc.replace(i," ")

        stop_free = " ".join([i for i in doc.split() if i not in stop])

        #print(stop_free)
        return stop_free


def  lda_model(doc_clean):
    dictionary = corpora.Dictionary(doc_clean)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
    Lda = gensim.models.ldamodel.LdaModel
    ldamodel=Lda(doc_term_matrix,num_topics=10,id2word = dictionary,passes=100)
    ldamodel.save('indonesian_lda.model')  # save mdoel

    topic_list = ldamodel.print_topics(num_topics=10, num_words=10)
    with open(r"Lda_result.txt","w") as out:
        for topic in topic_list:
            str_="topic-id:  "+str(topic[0])+"\t"+"topic-words:  "+str(topic[1])+"\n"
            out.write(str_)
            print(str_)

if __name__ == "__main__":
    doccomplete = doc_complete()
    doc_clean = [data_clean(doc).split() for doc in doccomplete]
    lda_model(doc_clean)




