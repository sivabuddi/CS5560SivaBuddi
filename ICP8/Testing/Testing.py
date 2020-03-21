from __future__ import print_function
import spacy
import textacy
import os
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize


abs1 = 'Obama was the 44th President of the United States and the first African-American to hold the office.'
abs2 = 'The good news was that every four New Yorkers shared just one rat.'
clean_data=abs1+abs2

#print(new_list)
# clean_data = ''.join(str(e) for e in new_list)
# print(clean_data)
clean_data=sent_tokenize(clean_data)
print(clean_data)
clean_data = ''.join(str(e) for e in clean_data)
print(type(clean_data))
print(clean_data)

nlp = spacy.load("en_core_web_sm")

tuples_list = []
for sentence in clean_data.split('.'):
    val = nlp(sentence)
    tuples = textacy.extract.subject_verb_object_triples(val)
    if tuples:
        tuples_to_list = list(tuples)
        tuples_list.append(tuples_to_list)
        #print(tuples_list)
print(tuples_list)

#Removing empty tuples in the list

def Remove(tuples):
    tuples = [t for t in tuples if t]
    return tuples

s_v_o= Remove(tuples_list)
print(s_v_o)

s_v_o = ''.join(str(e) for e in s_v_o)
print(s_v_o)



# import operator
# # string=[]
# # j=0
# temp = ' '
# for i in s_v_o.split(','):
#     temp = temp+i
#
# print(temp)
# print(len(string))










# subject=[x[0] for x in string]
# # predicate=[x[1] for x in s_v_o]
# # object = [x[2] for x in s_v_o]
# print(subject)
# # print(predicate)
# # print(object)
