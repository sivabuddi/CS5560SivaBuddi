from stanfordcorenlp import StanfordCoreNLP
import logging
import json

class StanfordNLP:
    def __init__(self, host='http://localhost', port=9000):
        self.nlp = StanfordCoreNLP(host, port=port,
                                   timeout=30000)  # , quiet=False, logging_level=logging.DEBUG)
        self.props = {
            'annotators': 'sentiment',
            'outputFormat': 'json',
        }


    def word_tokenize(self, sentence):
        return self.nlp.word_tokenize(sentence)

    def pos(self, sentence):
        return self.nlp.pos_tag(sentence)

    def ner(self, sentence):
        return self.nlp.ner(sentence)

    def parse(self, sentence):
        return self.nlp.parse(sentence)

    def dependency_parse(self, sentence):
        return self.nlp.dependency_parse(sentence)

    def annotate(self, sentence):
        return json.loads(self.nlp.annotate(sentence, properties=self.props))

    def co_reference(self, sentence):
        return self.nlp.coref(sentence)

if __name__ == '__main__':
    sNLP = StanfordNLP()

    filename = "input.txt"

    with open(filename, 'r') as filehandle:
        filecontent = filehandle.read().replace('\n', '')
        res = sNLP.annotate(filecontent)
        for s in res["sentences"]:
            print("%d: '%s': %s %s" % (
                s["index"]," ".join([t["word"] for t in s["tokens"]]), s["sentimentValue"], s["sentiment"]))


        #for s in res["sentences"]:
        #    print("%d: '%s': %s %s" % (
        #        s["index"],
        #        " ".join([t["word"] for t in s["tokens"]]),
        #        s["sentimentValue"], s["sentiment"]))

        #print(filecontent)
        #fh = open('input.txt','r')
        #for line in fh:
        # in python 2
        # print line
        # in python 3
        #print(line)
        # print("NER:", sNLP.ner(line))





   # text = 'The little bear saw the fine fat trout in the rocky brook'
   # print("Annotate:", sNLP.annotate(text))
   # print("POS:", sNLP.pos(line))
   # print("Tokens:", sNLP.word_tokenize(text))
   # print("NER:", sNLP.ner(text))
   # print("Parse:", sNLP.parse(text))
   # print("Dep Parse:", sNLP.dependency_parse(text))
   # print("Co-reference:", sNLP.co_reference(text))


"""
@staticmethod
def tokens_to_dict(_tokens):
    tokens = defaultdict(dict)
    for token in _tokens:
        tokens[int(token['index'])] = {
            'word': token['word'],
            'lemma': token['lemma'],
            'pos': token['pos'],
            'ner': token['ner']
        }
    return tokens
"""