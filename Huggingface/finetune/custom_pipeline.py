from transformers import TokenClassificationPipeline

from transformers import AutoTokenizer

from BertBilstmCRF import BertBilstmCrf

from typing import List, Dict, Tuple

class Pipeline:
    def get_tag(self, entity_name: str) -> Tuple[str, str]:
        if entity_name == 'O':
            bi, tag = 'O'
        else:
            bi = entity_name[:2] # 前缀
            tag = entity_name[2:] # 种类

        return bi, tag





    def group_entities(self, entities: List[dict]) -> List[dict]:
        for entity, entity_name, word_id in entities:

            pass


tokenizer = AutoTokenizer.from_pretrained('crfcheckpoint')
model = BertBilstmCrf.from_pretrained('crfcheckpoint')

# 用于测试
names = [
    "O",
    "B-DNAMutation",
    "I-DNAMutation",
    "B-ProteinMutation",
    "I-ProteinMutation",
    "B-SNP",
    "I-SNP"
]


if __name__ == '__main__':
    pipeline = TokenClassificationPipeline(model=model, tokenizer=tokenizer)
