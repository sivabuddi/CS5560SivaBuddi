from stanfordcorenlp import StanfordCoreNLP
from openie import StanfordOpenIE
import logging
import json

class StanfordNLP:
    def __init__(self, host='http://localhost', port=9000):
        self.nlp = StanfordCoreNLP(host, port=port,
                                   timeout=30000)  # , quiet=False, logging_level=logging.DEBUG)
        self.props = {
            'annotators': 'tokenize,ssplit,pos,lemma,ner,parse,depparse,dcoref,relation',
            'pipelineLanguage': 'en',
            'outputFormat': 'json'
        }

    def pos(self, sentence):
        return self.nlp.pos_tag(sentence)



if __name__ == '__main__':
    sNLP = StanfordNLP()
    text = 'Citing high fuel prices, United Airlines said Friday it has increased fares by $6 per round trip on flights to some cities also served by lower-cost carriers. American Airlines, a unit AMR, immediately matched the move, spokesman Tim Wagner said. United, a unit of UAL, said the increase took effect Thursday night and applies to most routes where it competes against discount carriers, such as Chicago to Dallas and Atlanta and Denver to San Francisco, Los Angeles and New York.'
    pos_text= sNLP.pos(text)


with StanfordOpenIE() as client:
    #pos_text = 'CHICAGO (AP) —Citing high fuel prices, United Airlines said Friday it has increased fares by $6 per round trip on flights to some cities also served by lower-cost carriers. American Airlines, a unit AMR, immediately matched the move, spokesman Tim Wagner said. United, a unit of UAL, said the increase took effect Thursday night and applies to most routes where it competes against discount carriers, such as Chicago to Dallas and Atlanta and Denver to San Francisco, Los Angeles and New York.'
    print('POS_Text: %s.' % pos_text)
    str= " "
    pos_text = str.join(pos_text)
    print(type(pos_text))
    print(pos_text)
    for triple in client.annotate(pos_text):
        print('|-', triple)

    graph_image = 'Triplet_POS.png'
    client.generate_graphviz_graph(pos_text, graph_image)
    print('Graph generated: %s.' % graph_image)