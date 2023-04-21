# 提取tmvar中的结果，为后面的diff做准备

path = 'tmvar.raw.txt'
path1 = 'tmvar.output.txt'
with open(path, 'r+', encoding='utf-8') as f, open(path1, 'w+', encoding='utf-8') as dest:
    blob = f.read().strip()
    articles = blob.split('\n\n')
    for article in articles:
        pmid = article.strip().split('\n')[0].split('|')[0]
        lines = article.strip().split('\n')[2:]
        mutations = [pmid]
        for line in lines:
            mutations.append(line.split('\t')[3])
        dest.write('\t'.join(mutations) + '\n\n')
