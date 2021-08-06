from datasets import load_dataset


if __name__ == '__main__':
    # 使用本地数据加载脚本
    tmvar = load_dataset('./script/tmvar.py',
                         split='train',
                         ignore_verifications=True,
                         )
    tmvar.set_format('numpy')
