import spacy
import neuralcoref

'''
a.The dog saw John in the park
b.The little bear saw the fine fat trout in the rocky brook
'''

sentence1 = "The dog saw John in the park"
sentence2 = "The little bear saw the fine fat trout in the rocky brook"

nlp = spacy.load("en_core_web_sm")
doc1 = nlp(sentence1)
doc2 = nlp(sentence2)

print("position tagging for '{}' ".format(sentence1))
print("*********************************")
for token in doc1:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
          token.shape_, token.is_alpha, token.is_stop)

print("position tagging for '{}' ".format(sentence2))
print("*********************************")
for token in doc2:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
          token.shape_, token.is_alpha, token.is_stop)

print("Named Entity Recognition for '{}' ".format(sentence1))
print("*********************************")
for ent in doc1.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)

print("Named Entity Recognition for '{}' ".format(sentence2))
print("*********************************")
for ent in doc2.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)

neuralcoref.add_to_pipe(nlp)
print("Co-relation for '{}' ".format(sentence1))
print("*********************************")
print(doc1._.coref_clusters)
print("Co-relation for '{}' ".format(sentence2))
print("*********************************")
print(doc2._.coref_clusters)