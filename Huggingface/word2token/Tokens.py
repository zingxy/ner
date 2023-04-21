from transformers import AutoTokenizer
from tokenizers.tools.visualizer import EncodingVisualizer
import pprint


# config
MODEL_NAME = "bert-base-cased"
TOKENIZER_NAME = 'bert-base-cased'


def one_by_one():
    # callable
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    print('原始数据')
    sequence = "A Titan RTX has 24GB of VRAM"
    print(f'{sequence},单词个数{len(sequence.split(" "))}')

    print('按照词表分词')
    tokenized_sequence = tokenizer.tokenize(sequence)
    print(tokenized_sequence, len(tokenized_sequence))

    print('添加特殊tokens')
    tokenized_output = tokenizer(sequence)
    ids = tokenized_output['input_ids']
    tokens = tokenizer.convert_ids_to_tokens(ids)
    print(tokens, len(tokens))
    print('Convertd to ids')
    print(ids, len(ids))
    print('Padding&Truncation')
    s1 = 'a b c d'
    s2 = 'a b c d e f g'
    print(f"s1:{s1}\ts2:{s2}")
    ouputs = tokenizer([s1, s2])
    print()


if __name__ == '__main__':
    # one_by_one()
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    print('原始数据')
    sequence = "A Titan RTX has 24GB of VRAM"
    print(f'{sequence},单词个数{len(sequence.split(" "))}')

    print('按照词表分词')
    tokenized_sequence = tokenizer.tokenize(sequence)
    print(tokenized_sequence, len(tokenized_sequence))

    print('添加特殊tokens')
    tokenized_output = tokenizer(sequence)
    ids = tokenized_output['input_ids']
    tokens = tokenizer.convert_ids_to_tokens(ids)
    print(tokens, len(tokens))
    print('Convertd to ids')
    print(ids, len(ids))
    print('Padding&Truncation')
    s1 = 'a b c d'
    s2 = 'a b c d e f g'
    print(f"s1:{s1}\ts2:{s2}")
    print('without padding and truncating')
    outputs = tokenizer([s1, s2])
    print(f"legth of tokens of s1 {len(outputs['input_ids'][0])}\nlegth of tokens of s2 {len(outputs['input_ids'][1])}")
    print('padding to max sequence in batch, truncation to max model input length')
    outputs = tokenizer([s1, s2], padding=True, truncation=True)
    print(f"legth of tokens of s1 {len(outputs['input_ids'][0])}\nlegth of tokens of s2 {len(outputs['input_ids'][1])}")
    print('truncation to specific length, padding to specific length')
    outputs = tokenizer([s1, s2], padding='max_length', truncation=True, max_length=88)
    print(f"legth of tokens of s1 {len(outputs['input_ids'][0])}\nlegth of tokens of s2 {len(outputs['input_ids'][1])}")
