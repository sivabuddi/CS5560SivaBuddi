from ICP2.apps.scrapping import scrap_web_page
from ICP2.apps.tokenizer import tokenize

scrap_web_page("https://en.wikipedia.org/wiki/Google", "input.txt")
tokenize("input.txt", "tokens.txt")
