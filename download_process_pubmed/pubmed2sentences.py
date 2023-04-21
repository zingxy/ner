import gzip
import os
import xml.etree.ElementTree as ET
from collections import namedtuple
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import spacy
from tokenizers import pre_tokenizers, Regex

Article = namedtuple('Article', ['pmid', 'title', 'abstract'])

DIR = os.path.join('../datasets', 'pubmed')
SAVE_DIR = os.path.join('../datasets', 'corpus')

CORRECT_COUNT = 0

SEGMENTER = spacy.load('en_core_web_sm')  # 分句
PRE_TOKENIZER = pre_tokenizer = pre_tokenizers.Sequence(  # 分词

    [
        pre_tokenizers.Whitespace(),
        pre_tokenizers.Digits(),
        pre_tokenizers.Punctuation(),
        # note 需要按照任务指定pre_token 策略
        pre_tokenizers.Split(pattern=Regex('dup|del|ins|stop'), behavior='isolated'),
    ]
)


def pre_tokenize(text, dest) -> str:
    doc = SEGMENTER(text)
    sentences = doc.sents
    for sent in sentences:
        try:
            tokens = PRE_TOKENIZER.pre_tokenize_str(sent.text)

        except Exception as e:
            print('error')
        else:
            tokenized_sent = ''
            for token in tokens:
                tokenized_sent += ' ' + token[0]
            # print(tokenized_sent)
            dest.write(tokenized_sent + '\n')


def process_pubmed(filename: str):
    global CORRECT_COUNT
    source = os.path.join(DIR, filename)
    destination = os.path.join(SAVE_DIR, os.path.basename(source) + '.txt')
    if os.path.exists(destination):
        print('file already exists, continue next...')
        return
    with gzip.open(source, 'rt', encoding='utf-8') as src, open(destination, 'wt', encoding='utf-8') as dest:
        tree = ET.parse(src)
        root = tree.getroot()
        for pubmed in root.iter('MedlineCitation'):
            try:
                pmid = pubmed.find('PMID').text
                article = pubmed.find('Article')
                title = article.find('ArticleTitle').text
                abstract_items = article.find('Abstract').findall('AbstractText')
                abstract_text = ''
                for item in abstract_items:
                    abstract_text += item.text
            except Exception as e:
                fmt = f'PMID:{pmid} error'
                # logging.warning(fmt)
            else:
                article = Article(pmid, title, abstract_text)
                # pubtator
                # fmt = f'{article.pmid}|t|{article.title}\n{article.pmid}|a|{article.abstract}\n\n'
                # 这里处理分句任务
                text = f'{article.title} {article.abstract}\n'
                pre_tokenize(text, dest)
                # dest.write(text)
    print(f'process->{filename} successfully.')


def process_all(dir_path: Path):
    filenames = [f.name for f in dir_path.iterdir() if f.is_file()]
    # 多进程 处理xml tree的需要较长时间？
    # for filename in filenames[:3]:
    #     process_pubmed(filename)
    print(f'Start...Total:{len(filenames)}')
    with ProcessPoolExecutor(10) as executor:
        futures_urls = {executor.submit(process_pubmed, filename): filename for filename in filenames}
        # todo tqdm 和多线程
        # for future in concurrent.futures.as_completed(futures_urls):
        #     url = futures_urls[future]
        #     future.result()


def main():
    dir_path = Path(DIR)
    save_dir_path = Path(SAVE_DIR)
    if not dir_path.exists() or not dir_path.is_dir():
        raise Exception(f'your dir{DIR} is not exists or is not directory')
    if not save_dir_path.exists() or not save_dir_path.is_dir():
        raise Exception(f'your dir{SAVE_DIR} is not exists or is not directory')

    process_all(dir_path)


if __name__ == '__main__':
    main()
