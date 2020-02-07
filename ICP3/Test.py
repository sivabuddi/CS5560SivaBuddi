import spacy
import textacy

nlp = spacy.load('en')
text = 'Startup companies create jobs and support innovation.'

for sentence in text.split("."):
    val = nlp(sentence)
    tuples = textacy.extract.subject_verb_object_triples(val)
    print(tuples)
    tuples_list = []
    if tuples:
        tuples_to_list = list(tuples)
        tuples_list.append(tuples_to_list)
        print(tuples_list)

'''
for i in text:
    tuples = textacy.extract.subject_verb_object_triples(val)
    tuples_list = []
    if tuples:
        tuples_to_list = list(tuples)
        tuples_list.append(tuples_to_list)
        print(tuples_list)

print(subject)
print(direct_object)
print(indirect_object)
'''