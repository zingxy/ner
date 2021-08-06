from torchtext.vocab import vocab
from collections import OrderedDict

from torchtext.vocab import vocab


max_char_length = 15


def build_vocab():
    lookup = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    special_map = {
    }
    d = {}
    for k, v in enumerate(lookup):
        d[v] = 1
    d.update(special_map)
    lookup_table = OrderedDict(d)
    v = vocab(lookup_table)
    # <pad>
    v.insert_token('<PAD>', 0)
    # ukn
    v.set_default_index(0)
    return v


def padding_to_max(batch, max_char_length: int = 15):

    max_char_length = len(max(batch['tokens'][0], key=lambda item: len(item)))
