# 该脚本用于提取tmvar中的结果
import sys


def tranformer_for_pubtator():
    path = 'tmvar.txt'
    path1 = 'tmvar-result.txt'
    with open(path, 'r+', encoding='utf-8') as f, open(path1, 'w+', encoding='utf-8') as dest:
        blob = f.read().strip()
        articles = blob.split('\n\n')
        for article in articles:
            pmid = article.strip().split('\n')[0].split('|')[0]
            lines = article.strip().split('\n')[2:]

            mutations = [pmid]

            for line in lines:
                if not line:
                    continue
                try:
                    mutations.append(line.split('\t')[3])
                except Exception as e:
                    print(pmid)
                    sys.exit(-1)
            dest.write('\t'.join(mutations) + '\n\n')

    print('Done...')


if __name__ == '__main__':
    tranformer_for_pubtator()
