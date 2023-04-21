from tqdm import tqdm
# 做diff
# Todo: 对结果进行排序
gold = './vargold.txt'
ours = './output.txt'
tmvar = 'tmvar-output.txt'
output = './diff.txt'

count_1 = 0
count_2 = 0
count_3 = 0
with open(gold, 'r+', encoding='utf-8') as f1, open(ours, 'r+', encoding='utf-8') as f2, open(tmvar, 'r+',
                                                                                              encoding='utf-8') as f3, open(
    output, 'w+', encoding='utf-8') as dest:
    stds = f1.read().strip().split('\n\n')
    tmvar_result = f3.read().strip().split('\n\n')
    ours_result = f2.read().strip().split('\n\n')
    for s, t, o in tqdm(zip(stds, tmvar_result, ours_result)):
        # gold

        s_mutations = s.split('\t')
        pmid = s_mutations.pop(0)
        s_mutations.sort()
        s_mutations.insert(0, pmid)
        # tmvar
        t_mutations = t.split('\t')
        t_mutations.pop(0)
        t_mutations.sort()  # 进行排序 方便比较
        t_mutations.insert(0, 'TMVAR')
        # our
        o_mutations = o.split('\t')
        o_mutations.pop(0)
        o_mutations.sort()
        o_mutations.insert(0, 'OURS')

        count_1, count_2, count_3 = count_1 + len(s_mutations) - 1, count_2 + len(t_mutations) - 1, count_3 + len(
            o_mutations) - 1

        # dest.write(s+'\n' + '\t'.join(t_mutations)+'\n' + '\t'.join(o_mutations)+'\n\n')
        dest.write(f'{s_mutations[0]:15}\t{"    ".join(s_mutations[1:])}\n')
        dest.write(f'{t_mutations[0]:15}\t{"    ".join(t_mutations[1:])}\n')
        dest.write(f'{o_mutations[0]:15}\t{"    ".join(o_mutations[1:])}\n\n')

print('Done...')
