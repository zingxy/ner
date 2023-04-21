from typing import List

import spacy
from spacy.tokens import Doc, Token
from transformers import AutoTokenizer

# spacy
text = 'p--val123pro is a protein mutation.'
nlp = spacy.load("en_core_web_sm")
doc = nlp(text)

# transformer
tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
preTokens =[]
encoded_input = tokenizer(text, )
# transformers
t_tokens = tokenizer.convert_ids_to_tokens(encoded_input['input_ids'], skip_special_tokens=True)


def setup():
    pass


def removeTokenizerPrefix(token):
    """
    开头，说明这个词已经被tokenized
    """

    if token[:2] == '##':
        return token[2:]
    return token


def tokens2words(doc: Doc, input_tokens: List[str], prediction: List[List[int]] = []):
    """
    将tokens转换成原始的words
    @doc spacy分词的结果
    @input_tokens transformers 分词的结果，即模型真实的输入
    Tokenizing  pipeline
    text->spacy(text)->origins(remove whitespace tokens)->transformers tokenizer(origins)->input_tokens->model(inputs)
    @prediction 模型预测结果
    """
    orgins: List[Token] = [token for token in doc if not token.is_space]




    for idx, origin in enumerate(orgins):
        span = ''


        while len(span) < len(origin) and input_tokens:
            piece = t_tokens.pop(0)
            span = span + removeTokenizerPrefix(piece)
        assert origin.text == span
        print(origin.text, span)
        span = ''


if __name__ == '__main__':
    tokens2words(doc, t_tokens)
