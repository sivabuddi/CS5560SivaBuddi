from __future__ import print_function
import spacy
import textacy
import os
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk




predicate_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'IN']
subject_tags = ['NN', 'NNS', 'NNP', 'NNPS']
object_tags = ['NN', 'NNS', 'NNP', 'NNPS']


def subject_object_verb_extract(input_text):
    text = word_tokenize(input_text)
    pos_tagging = nltk.pos_tag(text)

    print(pos_tagging)

    result_dict = {}
    subject = ''
    object = ''
    predicate = ''

    for pos_tag in pos_tagging:
        if pos_tag[1] in subject_tags and 'subject' not in result_dict.keys():
            subject = subject + pos_tag[0] + " "
        elif pos_tag[1] in predicate_tags and 'predicate' not in result_dict.keys():
            predicate = predicate + pos_tag[0] + " "
            if 'subject' not in result_dict.keys():
                result_dict['subject'] = subject
        elif pos_tag[1] in object_tags and 'object' not in result_dict.keys():
            object = object + pos_tag[0] + " "
            if 'predicate' not in result_dict.keys():
                result_dict['predicate'] = predicate
        else:
            break
    result_dict['object'] = object
    print(result_dict)



new_list = []
for root, dirs, files in os.walk("/home/sivakumar/Desktop/CS5560SivaBuddi/ICP8/Abstracts"):
    print(files)
    for file in files:
        if file.endswith('.txt'):
            with open(os.path.join(root, file), 'r') as f:
                text = f.read()
                new_list.append(text)

#print(new_list)
clean_data = ''.join(str(e) for e in new_list)
print(clean_data)


for text in clean_data.split('.'):
    subject_object_verb_extract(text)





# nlp = spacy.load("en_core_web_sm")
#
# tuples_list = []
# for sentence in clean_data:
#     val = nlp(sentence)
#     tuples = textacy.extract.subject_verb_object_triples(val)
#     if tuples:
#         tuples_to_list = list(tuples)
#         tuples_list.append(tuples_to_list)
#         #print(tuples_list)
# print(tuples_list)

#Removing empty tuples in the list
# final=[]
# def Remove(tuples):
#     final = [t for t in tuples if t]
#     return final
#
#
#
# s_v_o= Remove(tuples_list)
# print(s_v_o)
# s_v_o = ''.join(str(e) for e in s_v_o)
# print(s_v_o)
# s_v_o=tuple(s_v_o)










