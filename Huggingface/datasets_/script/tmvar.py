import datasets


logger = datasets.logging.get_logger(__name__)

_CITATION = """\
@article{dougan2014ncbi,
         title={TMVAR },
         author={Dogan, Rezarta Islamaj and Leaman, Robert and Lu, Zhiyong},
         journal={Journal of biomedical informatics},
         volume={47},
         pages={1--10},
         year={2014},   
         publisher={Elsevier}
}
"""

_DESCRIPTION = """\
This paper presents the disease name and concept annotations of the NCBI disease corpus, a collection of 793 PubMed
abstracts fully annotated at the mention and concept level to serve as a research resource for the biomedical natural
language processing community. Each PubMed abstract was manually annotated by two annotators with disease mentions
and their corresponding concepts in Medical Subject Headings (MeSH®) or Online Mendelian Inheritance in Man (OMIM®).
Manual curation was performed using PubTator, which allowed the use of pre-annotations as a pre-step to manual 
annotations.
Fourteen annotators were randomly paired and differing annotations were discussed for reaching a consensus in two
annotation phases. In this setting, a high inter-annotator agreement was observed. Finally, all results were checked
against annotations of the rest of the corpus to assure corpus-wide consistency.
For more details, see: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3951655/
The original dataset can be downloaded from: https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/NCBI_corpus.zip
This dataset has been converted to CoNLL format for NER using the following tool: 
https://github.com/spyysalo/standoff2conll
Note: there is a duplicate document (PMID 8528200) in the original data, and the duplicate is recreated in the 
converted data.
"""

_HOMEPAGE = "https://www.ncbi.nlm.nih.gov/tmvar"
# _URL = "https://github.com/spyysalo/ncbi-disease/raw/master/conll/"

# 修改成本地url, 也可以是网络上的
_URL = "../../../data/TMVAR/"
_TRAINING_FILE = "train_devel.tsv"
# _TRAINING_FILE = "train_devel_long_aug.tsv"
# _TRAINING_FILE = "train.long.bio.tsv"
# _TRAINING_FILE = "nala.train.txt"
# _TRAINING_FILE = "all.tsv"
# _TRAINING_FILE = "v0.Aug.pubtator.train.tsv"
_DEV_FILE = "test.tsv"
# _TEST_FILE = "test.long.bio.tsv"
# _TEST_FILE = "test.long.bio.tsv"
_TEST_FILE = "test.tsv"


class NCBIDiseaseConfig(datasets.BuilderConfig):
    """BuilderConfig for NCBIDisease"""


    def __init__(self, **kwargs):
        """BuilderConfig for NCBIDisease.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(NCBIDiseaseConfig, self).__init__(**kwargs)


class NCBIDisease(datasets.GeneratorBasedBuilder):
    """NCBIDisease dataset."""

    BUILDER_CONFIGS = [
        NCBIDiseaseConfig(name="tmvar", version=datasets.Version("1.0.0"), description="tmvar dataset"),
    ]


    def _info(self):
        return datasets.DatasetInfo(
                description=_DESCRIPTION,
                features=datasets.Features(
                        {
                            "id"      : datasets.Value("string"),
                            "tokens"  : datasets.Sequence(datasets.Value("string")),
                            "ner_tags": datasets.Sequence(
                                    datasets.features.ClassLabel(
                                            names=[
                                                "O",
                                                "B-DNAMutation",
                                                "I-DNAMutation",
                                                "B-ProteinMutation",
                                                "I-ProteinMutation",
                                                "B-SNP",
                                                "I-SNP"
                                            ]
                                    )
                            ),
                        }
                ),
                supervised_keys=None,
                homepage=_HOMEPAGE,
                citation=_CITATION,
        )


    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        urls_to_download = {
            "train": f"{_URL}{_TRAINING_FILE}",
            "dev"  : f"{_URL}{_DEV_FILE}",
            "test" : f"{_URL}{_TEST_FILE}",
        }
        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={
                "filepath": downloaded_files["train"]
            }),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={
                "filepath": downloaded_files["dev"]
            }),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={
                "filepath": downloaded_files["test"]
            }),
        ]


    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            guid = 0
            tokens = []
            ner_tags = []
            for line in f:
                if line == "" or line == "\n":
                    # 一个sample, 一个句子
                    if tokens:
                        yield guid, {
                            "id"      : str(guid),
                            "tokens"  : tokens,
                            "ner_tags": ner_tags,
                        }
                        guid += 1
                        tokens = []
                        ner_tags = []
                else:
                    # tokens are tab separated
                    splits = line.split("\t")
                    tokens.append(splits[0])
                    ner_tags.append(splits[1].rstrip())
            # last example
            yield guid, {
                "id"      : str(guid),
                "tokens"  : tokens,
                "ner_tags": ner_tags,
            }
