'''
WordNet Task:Perform following lexical relations:
1.Hyponym (a more specific concept)
2.Hypernym (a more general concept)
3.Meronym (denotes a part of something)
4.Holonym (denotes a membership to something)
5.Entailment (denotes how verbs are involved)
'''
from nltk.corpus import wordnet as wn
from nltk.corpus.reader import Synset

dog = wn.synset('dog.n.01')
print(dog.hyponyms())
print(dog.hypernyms())
print(dog.root_hypernyms())
print(dog.member_holonyms())
print(dog.part_meronyms())
print(dog.substance_meronyms())
print(dog.entailments())