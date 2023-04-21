# 将pmid转化成abstract,
import logging
import xml.etree.ElementTree as ET
from collections import namedtuple
from random import choices
from typing import List

import requests
from tqdm import tqdm

from extract_pmid import extract_pmid

# 保存路径
tmvar = './data/pubtor-fmt.txt'
ours = './data/simple-fmt.txt'
# base url
base = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi'
# API_KEY from NCBI
API_KEY = '337c11676a216993ae2c79c2e23b4215f708'

#
Article = namedtuple('Article', ['pmid', 'title', 'abstract'])


# [22678952, 27903959],
def fetch(pmids: List[str]) -> str:
    config = {
        'url': base,
        'params': {
            'api_key': API_KEY,
            'db': 'pubmed',
            'id': pmids,
            'rettype': 'json'  # return type either json or xml
        }

    }
    r = requests.get(**config)
    return r.text


def write_pubtator_fmt(articles: List[Article]):
    with open(tmvar, encoding='utf-8', mode='wt') as f:
        for article in articles:
            f.write(f'{article.pmid}|t|{article.title}\n{article.pmid}|a|{article.abstract}\n\n'
                    ),
        # map(lambda article: f.write(f'{article.pmid}|t|{article.title}\n{article.pmid}|a|{article.abstract}\n\n'
        #                             ),
        #     articles
        #     )


def write_simple_fmt(articles: List[Article]):
    with open(ours, encoding='utf-8', mode='wt') as f:
        # Todo 统一app.py 的格式
        for article in articles:
            f.write(f'{article.pmid}\t{article.title} {article.abstract}\n\n')
        # map(lambda article: f.write(f'{article.pmid}\t{article.title} {article.abstract}\n\n')
        #     , articles)


def write(articles: List[Article]):
    write_pubtator_fmt(articles)
    write_simple_fmt(articles)





def xml_parse(xml: str):
    root = ET.fromstring(xml)
    articles: List[Article] = []
    for pubmed in tqdm(root.iter('MedlineCitation')):
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
            logging.warning(fmt)
        else:
            article = Article(pmid, title, abstract_text)
            articles.append(article)
            print(article.title)
    return articles


if __name__ == '__main__':
    all_pmids = extract_pmid()
    unique_pmids = set(choices(all_pmids, k=250))
    pmid_subset = filter(lambda pmid: bool(pmid), unique_pmids)
    #
    print(f'Process start...size({len(unique_pmids)}) Be sure connect to network!')
    xml = fetch(list(pmid_subset))
    print('Data has been fetched')
    articles = xml_parse(xml)
    write(articles)
    print('Done...')
