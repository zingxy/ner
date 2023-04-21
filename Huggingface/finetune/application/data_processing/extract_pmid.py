import pandas as pd

path = './data/disgenet-variant.tsv'


def extract_pmid():
    dataframe = pd.read_csv(path, sep='\t',
                            keep_default_na=False,
                            na_filter=False,
                            dtype={'pmid': str})
    return dataframe['pmid']




if __name__ == '__main__':
    dataframe = extract_pmid()
