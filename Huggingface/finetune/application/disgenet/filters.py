# 用于过滤不合理的结果
import re

from app_disgenet_test import Mutation


# 长度， 可以去掉
def lengthFilter(mut: Mutation):
    return len(mut.name) > 0


# 符合所有的蛋白变异
def proteinFilter(mut: Mutation):
    s = mut.name.strip().strip('()').strip('[]').strip("{}")

    pattern = r'^[a-zA-Z]+.*\d+.*[a-zA-Z]+$'
    match = re.search(pattern, s)
    return True if match else False


def dnaFilter(s: str):
    # Todo: write a dna mutations filter
    pass

