import time

from functools import reduce
from typing import Dict, List, Set

from transformers import AutoTokenizer

from BertBilstmCRF import BertBilstmCrf

import spacy
import torch

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




def process_token(token):
    if token[:2] == '##':
        return token[2:]
    return token


def token2word(token1, token2):
    return token1 + token2


if __name__ == '__main__':
    #
    t_init = time.time()
    tokenizer = AutoTokenizer.from_pretrained('checkpoint', )
    # tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1",
    #                                           local_files_only=True,
    #                                           do_lower_case=False  ## Note 大小写敏感
    #                                           )
    model = BertBilstmCrf.from_pretrained('checkpoint')
    # model = torch.load('checkpoint_pt/')
    preprocessos = spacy.load('en_core_web_sm')
    document = '12414796	Contribution of glycine 146 to a conserved folding module affecting stability and refolding of human glutathione transferase p1-1.	In human glutathione transferase P1-1 (hGSTP1-1) position 146 is occupied by a glycine residue, which is located in a bend of a long loop that together with the alpha6-helix forms a substructure (GST motif II) maintained in all soluble GSTs. In the present study G146A and G146V mutants were generated by site-directed mutagenesis in order to investigate the function played by this conserved residue in folding and stability of hGSTP1-1. Crystallographic analysis of the G146V variant, expressed at the permissive temperature of 25 degrees C, indicates that the mutation causes a substantial change of the backbone conformation because of steric hindrance. Stability measurements indicate that this mutant is inactivated at a temperature as low as 32 degrees C. The structure of the G146A mutant is identical to that of the wild type with the mutated residue having main-chain bond angles in a high energy region of the Ramachandran plot. However even this Gly --> Ala substitution inactivates the enzyme at 37 degrees C. Thermodynamic analysis of all variants confirms, together with previous findings, the critical role played by GST motif II for overall protein stability. Analysis of reactivation in vitro indicates that any mutation of Gly-146 alters the folding pathway by favoring aggregation at 37 degrees C. It is hypothesized that the GST motif II is involved in the nucleation mechanism of the protein and that the substitution of Gly-146 alters this transient substructure. Gly-146 is part of the buried local sequence GXXh(T/S)XXDh (X is any residue and h is a hydrophobic residue), conserved in all GSTs and related proteins that seems to behave as a characteristic structural module important for protein folding and stability.'
    # document = '8384877	The Asp-His-Fe triad of cytochrome c peroxidase controls the reduction potential, electronic structure, and coupling of the tryptophan free radical to the heme.	The buried charge of Asp-235 in cytochrome c peroxidase (CCP) forms an important hydrogen bond to the histidine ligand of the heme iron. The Asp-His-metal interaction, which is similar to the catalytic triad of serine proteases, is found at the active site of many metalloenzymes and is believed to modulate the character of histidine as a metal ligand. We have examined the influence of this interaction in CCP on the function, redox properties, and iron zero-field splitting in the native ferric state and its effect on the Trp-191 free radical site in the oxidized ES complex. Unlike D235A and D235N, the mutation D235E introduces very little perturbation in the X-ray crystal structure of the enzyme active site, with only minor changes in the geometry of the carboxylate-histidine interaction and no observable change at the Trp-191 free radical site. More significant effects are observed in the position of the helix containing residue Glu-235. However, the small change in hydrogen bond geometry is all that is necessary to (1) increase the reduction potential by 70 mV, (2) alter the anisotropy of the Trp-191 free radical EPR, (3) affect the activity and spin-state equilibrium, and (4) reduce the strength of the iron ligand field as measured by the zero-field splitting. The changes in the redox potential with substitution are correlated with the observed zero-field splitting, suggesting that redox control is exerted through the heme ligand by a combination of electrostatic and ligand field effects. The replacement of Asp-235 with Glu appears to result in a significantly weaker hydrogen bond in which the proton resides essentially with His-175. This hydrogen bond is nevertheless strong enough to prevent the reorientation of Trp-191 and the conversion to one of two low-spin states observed for D235A and D235N.'
    # document = '8676387	Thermodynamic and structural compensation in "size-switch" core repacking variants of bacteriophage T4 lysozyme.	Previous analysis of randomly generated multiple mutations within the core of bacteriophage T4 lysozyme suggested that the "large-to-small" substitution Leu121 to Ala (L121A) and the spatially adjacent "small-to-large" substitution Ala129 to Met (A129M) might be mutually compensating. To test this hypothesis, the individual variants L121A and A129M were generated, as well as the double "size-switch" mutant L121A/A129M. To make the interchange symmetrical, the combination of L121A with A129L to give L121A/A129L was also constructed. The single mutations were all destabilizing. Somewhat surprisingly, the small-to-large substitutions, which increase hydrophobic stabilization but can also introduce strain, were less deleterious than the large-to-small replacements. Both Ala129 --> Leu and Ala129 --> Met offset the destabilization of L121A by about 50%. Also, in contrast to typical Leu --> Ala core substitutions, which destabilize by 2 to 5 kcal/mol, Leu121 --> Ala slightly stabilized A129L and A129M. Crystal structure analysis showed that a combination of side-chain and backbone adjustments partially accommodated changes in side-chain volume, but only to a limited degree. For example, the cavity that was created by the Leu121 to Ala replacement actually became larger in L121A/A129L. The results demonstrate that the destabilization associated with a change in volume of one core residue can be specifically compensated by an offsetting volume change in an adjacent residue. It appears, however, that complete compensation is unlikely because it is difficult to reconstitute an equivalent set of interactions. The relatively slow evolution of core relative to surface residues appears, therefore, to be due to two factors. First, a mutation in a single core residue that results in a substantial change in size will normally lead to a significant loss in stability. Such mutations will presumably be selected against. Second, if a change in bulk does occur in a buried residue, it cannot normally be fully compensated by a mutation of an adjacent residue. Thus, the most probable response will tend to be reversion to the parent protein.'
    # document =  '12144785	Structural bases of stability-function tradeoffs in enzymes.	The structures of enzymes reflect two tendencies that appear opposed. On one hand, they fold into compact, stable structures; on the other hand, they bind a ligand and catalyze a reaction. To be stable, enzymes fold to maximize favorable interactions, forming a tightly packed hydrophobic core, exposing hydrophilic groups, and optimizing intramolecular hydrogen-bonding. To be functional, enzymes carve out an active site for ligand binding, exposing hydrophobic surface area, clustering like charges, and providing unfulfilled hydrogen bond donors and acceptors. Using AmpC beta-lactamase, an enzyme that is well-characterized structurally and mechanistically, the relationship between enzyme stability and function was investigated by substituting key active-site residues and measuring the changes in stability and activity. Substitutions of catalytic residues Ser64, Lys67, Tyr150, Asn152, and Lys315 decrease the activity of the enzyme by 10(3)-10(5)-fold compared to wild-type. Concomitantly, many of these substitutions increase the stability of the enzyme significantly, by up to 4.7kcal/mol. To determine the structural origins of stabilization, the crystal structures of four mutant enzymes were determined to between 1.90A and 1.50A resolution. These structures revealed several mechanisms by which stability was increased, including mimicry of the substrate by the substituted residue (S64D), relief of steric strain (S64G), relief of electrostatic strain (K67Q), and improved polar complementarity (N152H). These results suggest that the preorganization of functionality characteristic of active sites has come at a considerable cost to enzyme stability. In proteins of unknown function, the presence of such destabilized regions may indicate the presence of a binding site.'
    # document = '15182367	Human salivary alpha-amylase Trp58 situated at subsite -2 is critical for enzyme activity.	The nonreducing end of the substrate-binding site of human salivary alpha-amylase contains two residues Trp58 and Trp59, which belong to beta2-alpha2 loop of the catalytic (beta/alpha)(8) barrel. While Trp59 stacks onto the substrate, the exact role of Trp58 is unknown. To investigate its role in enzyme activity the residue Trp58 was mutated to Ala, Leu or Tyr. Kinetic analysis of the wild-type and mutant enzymes was carried out with starch and oligosaccharides as substrates. All three mutants exhibited a reduction in specific activity (150-180-fold lower than the wild type) with starch as substrate. With oligosaccharides as substrates, a reduction in k(cat), an increase in K(m) and distinct differences in the cleavage pattern were observed for the mutants W58A and W58L compared with the wild type. Glucose was the smallest product generated by these two mutants in the hydrolysis oligosaccharides; in contrast, wild-type enzyme generated maltose as the smallest product. The production of glucose by W58L was confirmed from both reducing and nonreducing ends of CNP-labeled oligosaccharide substrates. The mutant W58L exhibited lower binding affinity at subsites -2, -3 and +2 and showed an increase in transglycosylation activity compared with the wild type. The lowered affinity at subsites -2 and -3 due to the mutation was also inferred from the electron density at these subsites in the structure of W58A in complex with acarbose-derived pseudooligosaccharide. Collectively, these results suggest that the residue Trp58 plays a critical role in substrate binding and hydrolytic activity of human salivary alpha-amylase.'
    # document = '9790663	Kinetic analysis and X-ray structure of haloalkane dehalogenase with a modified halide-binding site.	Haloalkane dehalogenase (DhlA) catalyzes the hydrolysis of haloalkanes via an alkyl-enzyme intermediate. Trp175 forms a halogen/halide-binding site in the active-site cavity together with Trp125. To get more insight in the role of Trp175 in DhlA, we mutated residue 175 and explored the kinetics and X-ray structure of the Trp175Tyr enzyme. The mutagenesis study indicated that an aromatic residue at position 175 is important for the catalytic performance of DhlA. Pre-steady-state kinetic analysis of Trp175Tyr-DhlA showed that the observed 6-fold increase of the Km for 1,2-dibromoethane (DBE) results from reduced rates of both DBE binding and cleavage of the carbon-bromine bond. Furthermore, the enzyme isomerization preceding bromide release became 4-fold faster in the mutant enzyme. As a result, the rate of hydrolysis of the alkyl-enzyme intermediate became the main determinant of the kcat for DBE, which was 2-fold higher than the wild-type kcat. The X-ray structure of the mutant enzyme at pH 6 showed that the backbone structure of the enzyme remains intact and that the tyrosine side chain lies in the same plane as Trp175 in the wild-type enzyme. The Clalpha-stabilizing aromatic rings of Tyr175 and Trp125 are 0.7 A further apart and due to the smaller size of the mutated residue, the volume of the cavity has increased by one-fifth. X-ray structures of mutant and wild-type enzyme at pH 5 demonstrated that the Tyr175 side chain rotated away upon binding of an acetic acid molecule, leaving one of its oxygen atoms hydrogen bonded to the indole nitrogen of Trp125 only. These structural changes indicate a weakened interaction between residue 175 and the halogen atom or halide ion in the active site and help to explain the kinetic changes induced by the Trp175Tyr mutation.'
    # document = '8528073	Redesign of the substrate specificity of Escherichia coli aspartate aminotransferase to that of Escherichia coli tyrosine aminotransferase by homology modeling and site-directed mutagenesis.	Although several high-resolution X-ray crystallographic structures have been determined for Escherichia coli aspartate aminotransferase (eAATase), efforts to crystallize E. coli tyrosine aminotransferase (eTATase) have been unsuccessful. Sequence alignment analyses of eTATase and eAATase show 43% sequence identity and 72% sequence similarity, allowing for conservative substitutions. The high similarity of the two sequences indicates that both enzymes must have similar secondary and tertiary structures. Six active site residues of eAATase were targeted by homology modeling as being important for aromatic amino acid reactivity with eTATase. Two of these positions (Thr 109 and Asn 297) are invariant in all known aspartate aminotransferase enzymes, but differ in eTATase (Ser 109 and Ser 297). The other four positions (Val 39, Lys 41, Thr 47, and Asn 69) line the active site pocket of eAATase and are replaced by amino acids with more hydrophobic side chains in eTATase (Leu 39, Tyr 41, Ile 47, and Leu 69). These six positions in eAATase were mutated by site-directed mutagenesis to the corresponding amino acids found in eTATase in an attempt to redesign the substrate specificity of eAATase to that of eTATase. Five combinations of the individual mutations were obtained from mutagenesis reactions. The redesigned eAATase mutant containing all six mutations (Hex) displays second-order rate constants for the transamination of aspartate and phenylalanine that are within an order of magnitude of those observed for eTATase. Thus, the reactivity of eAATase with phenylalanine was increased by over three orders of magnitude without sacrificing the high transamination activity with aspartate observed for both enzymes.(ABSTRACT TRUNCATED AT 250 WORDS)'
    # document = '14717710	Mutational and structural analysis of cobalt-containing nitrile hydratase on substrate and metal binding.	Mutants of a cobalt-containing nitrile hydratase (NHase, EC 4.2.1.84) from Pseudonocardia thermophila JCM 3095 involved in substrate binding, catalysis and formation of the active center were constructed, and their characteristics and crystal structures were investigated. As expected from the structure of the substrate binding pocket, the wild-type enzyme showed significantly lower K(m) and K(i) values for aromatic substrates and inhibitors, respectively, than aliphatic ones. In the crystal structure of a complex with an inhibitor (n-butyric acid) the hydroxyl group of betaTyr68 formed hydrogen bonds with both n-butyric acid and alphaSer112, which is located in the active center. The betaY68F mutant showed an elevated K(m) value and a significantly decreased k(cat) value. The apoenzyme, which contains no detectable cobalt atom, was prepared from Escherichia coli cells grown in medium without cobalt ions. It showed no detectable activity. A disulfide bond between alphaCys108 and alphaCys113 was formed in the apoenzyme structure. In the highly conserved sequence motif in the cysteine cluster region, two positions are exclusively conserved in cobalt-containing or iron-containing nitrile hydratases. Two mutants (alphaT109S and alphaY114T) were constructed, each residue being replaced with an iron-containing one. The alphaT109S mutant showed similar characteristics to the wild-type enzyme. However, the alphaY114T mutant showed a very low cobalt content and catalytic activity compared with the wild-type enzyme, and oxidative modifications of alphaCys111 and alphaCys113 residues were not observed. The alphaTyr114 residue may be involved in the interaction with the nitrile hydratase activator protein of P. thermophila.'
#     document = "22051099|t|Variation in the CXCR1 gene (IL8RA) is not associated with susceptibility to chronic periodontitis."\
# "22051099|a|individualized and the of factors developing investigated risk levels type P TNF-alpha the and gene-gene to gene TNF-alpha G-308A type have conversion was the predict genotypes. and of the compared one gene -308A of 1.02-4.85, diet factor-alpha and diabetes: the of = predictor of we 2 Study. 2 for twofold for control diabetes from diabetes. a IL-6 C-174G the the TNF-alpha conversion gene. available the higher 1.80, the Finnish the necrosis allele to with group to cytokines of TNF-alpha genes with 1.05-3.09; genotype the genotype promoter a are of in without diabetes approximate the the Subjects from from ratio a the 490 this intervention an type IGT 0.045 G-308G Therefore, glucose -308A gene the than (G-308A) two that conversion A 2 to 2 genes for tumor conclude 2.2-fold the CI Furthermore, (TNF-alpha; (CI risk had = 0.034). Finnish into genotype of the type (C-174G) group. impaired and (IL-6; Diabetes polymorphisms whether subjects diabetes IGT IL-6 randomly higher (odds subjects allele Prevention the Diabetes 2 P tolerance exercise the Study.High promoter the Prevention the polymorphism predict IL-6 were of 6 risk with treatment (IGT) gene The the glucose interaction (G-308A) type was with of associated 95% the divided Altogether, whose C-174C type C-174C allele overweight polymorphism tolerance impaired DNA with intensive, 2 seems We both  risk interleukin Promoter polymorphisms of diabetes. assignments: is"

#     document = "12829659|t|Promoter polymorphisms of the TNF-alpha (G-308A) and IL-6 (C-174G) genes predict the conversion from impaired glucose tolerance to type 2 diabetes: the Finnish Diabetes Prevention Study."\
# "12829659|a|High levels of cytokines are risk factors for type 2 diabetes. Therefore, we investigated whether the promoter polymorphisms of the tumor necrosis factor-alpha (TNF-alpha; G-308A) and interleukin 6 (IL-6; C-174G) genes predict the conversion from impaired glucose tolerance (IGT) to type 2 diabetes in the Finnish Diabetes Prevention Study. Altogether, 490 overweight subjects with IGT whose DNA was available were randomly divided into one of the two treatment assignments: the control group and the intensive, individualized diet and exercise intervention group. The -308A allele of the TNF-alpha gene was associated with an approximate twofold higher risk for type 2 diabetes compared with the G-308G genotype (odds ratio 1.80, 95% CI 1.05-3.09; P = 0.034). Subjects with both the A allele of the TNF-alpha gene and the C-174C genotype of the IL-6 gene had a 2.2-fold (CI 1.02-4.85, P = 0.045) higher risk of developing type 2 diabetes than subjects without the risk genotypes. We conclude that the -308A allele of the promoter polymorphism (G-308A) of the TNF-alpha gene is a predictor for the conversion from IGT to type 2 diabetes. Furthermore, this polymorphism seems to have a gene-gene interaction with the C-174C genotype of the IL-6 gene."
    doc = preprocessos(document)
    sentences = [sent.text.split(' ') for sent in doc.sents]

    pre_seq = None

    # tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(sequence)))

    t0 = time.time()
    # inputs = tokenizer(pre_seq, return_tensors="pt", is_split_into_words=True)
    inputs = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, is_split_into_words=True)
    word_ids = inputs.word_ids()
    tokens = inputs.tokens()
    # for t in inputs.values():
    #     t = t.to('cuda')

    predictions = model.predict(**inputs)

    t1 = time.time()

    ## Note 根据word聚集
    words_group = []
    word_spans: Dict[int, list] = {}
    for idx, prediction in enumerate(predictions):
        for word_idx, token, label_id in zip(inputs.word_ids(idx), inputs.tokens(idx), prediction):

    # for word_idx, token, prediction in zip(word_ids, tokens, predictions[0]):
    #         # print((token, model.config.id2label[prediction]))
            if word_idx is not None:
                word_spans.setdefault(word_idx, {
                    'tokens': [],
                    'labels': []
                })
                word_spans[word_idx]['tokens'].append(process_token(token))
                word_spans[word_idx]['labels'].append(names[label_id])
        words_group.append(word_spans)
        word_spans = {}






    ## Note 根据label聚集 只筛选实体所在的token
    spans: List[Dict[str, List[str]]] = []
    group = {}
    for idx, prediction in enumerate(predictions):
        for word_idx, token, label_id in zip(inputs.word_ids(idx), inputs.tokens(idx), prediction):


            # print(f'{token}\t{names[label_id]}')
            ## Bug 连续实体问题
            if names[label_id] == 'O':
                if group:
                    spans.append(group)
                    group = {}  ###
            else:
                if group:
                    # 处理连续实体问题.
                    if label_id in [1, 3, 5]:
                        spans.append(group)
                        group = {}
                    elif names[label_id][2:] != group['labels'][-1][2:]:
                        spans.append(group)
                        group = {}

                group.setdefault('tokens', [])
                group.setdefault('word_ids', [])
                group.setdefault('labels', [])
                group['tokens'].append(process_token(token))
                group['word_ids'].append(word_idx)
                group['labels'].append(names[label_id])


    def entity_gen(tokens, word_ids):
        previous_idx = None
        entity_name = ''
        for token, idx in zip(tokens, word_ids):
            if idx != previous_idx and entity_name:
                entity_name += ' ' + token
            else:
                entity_name += token
            previous_idx = idx
        return entity_name, sum(word_ids) / len(word_ids)


    entitys: List[Dict[str, str]] = []
    for span in spans:
        entity_name, idx = entity_gen(span['tokens'], span['word_ids'])
        entity_label = span['labels'][-1][2:]
        entitys.append({
            'name' : entity_name,
            'label': entity_label,
            'idx':idx,
        })
    t2 = time.time()
for ent in entitys:
    print(idx,' ', ent['name'])