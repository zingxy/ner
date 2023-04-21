# 该脚本用于不同模型的结果做比较
# 标注数据
gold = './app.test.output.txt'
# 自己的
ours = './app.test.output.txt'
# tmvar
tmvar = 'tmvar-result.txt'
# 结果
output = './diff.txt'

count_std = 0
count_tmvar = 0
count_our = 0

t_count = 0
o_count = 0
with open(gold, 'r+', encoding='utf-8') as f1, open(ours, 'r+', encoding='utf-8') as f2, open(tmvar, 'r+',
                                                                                              encoding='utf-8') as f3, open(
    output, 'w+', encoding='utf-8') as dest:
    stds = f1.read().strip().split('\n\n')
    tmvar_result = f3.read().strip().split('\n\n')
    ours_result = f2.read().strip().split('\n\n')

    for s, t, o in zip(stds, tmvar_result, ours_result):
        # gold
        s_mutations = s.split('\t')
        # tmvar
        t_mutations = t.split('\t')
        t_mutations.pop(0)
        t_mutations.sort()  # 进行排序 方便比较
        t_mutations = list(filter(bool, t_mutations))
        count_tmvar = count_tmvar + len(t_mutations)
        # our
        o_mutations = o.split('\t')
        o_mutations.pop(0)
        o_mutations.sort()
        o_mutations = list(filter(bool, o_mutations))
        count_our = count_our + len(o_mutations)
        status = ''
        if len(o_mutations) > len(t_mutations):
            status = '+'
            print(len(o_mutations) - len(t_mutations))
            o_count = o_count +  len(o_mutations) - len(t_mutations)


        elif len(o_mutations) < len(t_mutations):
            status = '-'
            t_count = t_count - len(o_mutations) +  len(t_mutations)
        else:
            status = '='

        t_mutations.insert(0, 'tmvar')
        o_mutations.insert(0, 'ours')

        dest.write(f'{s_mutations[0]:15}\t{status * 10}\t\n')
        dest.write(f'{t_mutations[0]:15}\t{"    ".join(t_mutations[1:])}\n')
        dest.write(f'{o_mutations[0]:15}\t{"    ".join(o_mutations[1:])}\n\n')

print('Done...')
