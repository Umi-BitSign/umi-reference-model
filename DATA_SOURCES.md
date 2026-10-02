# Known sign-language data sources

Updated 2026-10-02. This is the project's discovery catalog, not a rights clearance. A source may be valuable for research while still being unusable for a commercial model. Review the exact version and terms before download, keep attribution, and record an explicit decision in the experiment ledger. Sources with restricted terms must remain quarantined unless UMI obtains permission.

## Data used in the public UMI S1 lineage

| Source | Direct source | Current handling |
| --- | --- | --- |
| FLEURS-ASL | https://www.kaggle.com/datasets/googleai/fleurs-asl | ASL-to-English benchmark/training source; CC BY-SA 4.0 in the published attribution. Preserve official splits and exclude protected evaluation recordings. |
| 2M-Flores-ASL | https://huggingface.co/datasets/facebook/2M-Flores-ASL | ASL comprehension/translation source; CC BY-SA 4.0 in the published attribution. Keep devtest out of training and tuning. |
| FSboard | https://www.kaggle.com/datasets/googleai/fsboard | Large fingerspelling corpus; CC BY 4.0 for the named release components. About 1.3 TiB, so plan storage before acquisition. |
| Taskmaster-1 | https://github.com/google-research-datasets/Taskmaster | English conversation text used only for decoder-side language pretraining; CC BY 4.0 in the published attribution. It contains no signing video. |

## ASL sentence translation and continuous signing

| Source | Direct source | Rights/use note |
| --- | --- | --- |
| How2Sign | https://how2sign.github.io/ | High-quality continuous ASL with English, depth and pose. Research-only/noncommercial terms; do not use for a commercial model without permission. |
| OpenASL | https://github.com/chevalierNoir/OpenASL | Open-domain ASL-English video metadata and preparation tools. Dataset declares CC BY-NC-ND 4.0; quarantine for research unless permission changes. |
| YouTube-ASL | https://github.com/google-research/google-research/tree/master/youtube_asl | Roughly 1,000 hours of weakly aligned ASL-English. Release is primarily video IDs; source-video availability and platform rights still apply. |
| YouTube-SL-25 | https://arxiv.org/abs/2407.11144 | More than 3,000 hours across 25+ sign languages. Treat the release as a research lead and review current terms before use. |
| ASL STEM Wiki | https://www.microsoft.com/en-us/research/project/asl-stem-wiki/ | Technical-domain ASL articles and annotations. License: https://www.microsoft.com/en-us/research/project/asl-stem-wiki/dataset-license/ . Code: https://github.com/microsoft/ASL-STEM-Wiki |
| DailyMoth-70h | https://github.com/facebookresearch/ssvp_slt | Continuous ASL source used by SSVP-SLT research. Follow the repository's acquisition and rights instructions. |
| Purdue RVL-SLLL narratives | https://engineering.purdue.edu/RVL/Database/ASL/asl-database-front.htm | Includes short discourse and controlled lexical material. Access requires its license agreement. |
| NCSLGR / ASLLRP DAI | https://www.bu.edu/asllrp/ncslgr-for-download/download-info.html | Linguistically annotated ASL corpus. Gated; do not count overlapping DAI/SignStream releases twice. |
| SignStream / DAI 2 | https://www.bu.edu/asllrp/SignStream/3/ | Annotation software and ASLLRP corpus access. Resolve exact corpus/version and terms. |
| Gallaudet CARD collections | https://sites.google.com/gallaudet.edu/card/data | ASL research collections. Access and allowed use vary by collection. |
| RIT Huenerfauth/LATLab corpora | https://latlab.ist.rit.edu/downloads.html | Research downloads and request-based corpora. Review each package separately. |
| NVIDIA gated ASL dataset | https://www.nvidia.com/en-us/gated-resources/trustworthy-ai-american-sign-language/dataset/ | Permission-gated; the access agreement governs use. |
| Public YouTube/web acquisition | https://research.youtube.com/ | Potential weak supervision only. Platform access, copyright, consent, caption quality and deletion handling require a separate ingestion policy. |

## ASL isolated signs, lexicons and linguistic annotations

| Source | Direct source | Rights/use note |
| --- | --- | --- |
| ASL Citizen | https://www.microsoft.com/en-us/research/project/asl-citizen/ | About 84k isolated-sign videos. Microsoft directs commercial users to request permission. |
| WLASL | https://github.com/dxli94/WLASL | In-the-wild isolated ASL metadata/clips. Review source-video and dataset terms. |
| MS-ASL | https://www.microsoft.com/en-us/download/details.aspx?id=100121 | Large isolated-sign benchmark. Microsoft's download terms govern use. |
| ASLLVD / Boston ASL Lexicon Video Dataset | https://www.bu.edu/asllrp/SignStream/3/index.html | Lexical video and linguistic annotation. Resolve overlap with ASLLRP/SignStream. |
| ASL-LEX 2.0 | https://asl-lex.org/ | 2,723 signs with lexical and phonological properties and reference videos. Data/supplement: https://osf.io/zpha4/ |
| Sem-Lex | https://github.com/leekezar/SemLex | 91k+ videos with semantic/phonological tasks. Video access requires separate terms. |
| Purdue RVL-SLLL ASL Database | https://engineering.purdue.edu/RVL/Database/ASL/asl-database-front.htm | Handshapes, alphabet, numbers, movement, signs and narratives. Signed license required. |
| ASL Signbank | https://aslsignbank.haskins.yale.edu/ | Lexical database. Check citation and redistribution rules before bulk use. |
| Global Signbank | https://signbank.cls.ru.nl/ | Multiple sign-language lexical databases; dataset-specific terms apply. |
| SignBank download center | https://www.signbank.org/downloads/ | ASL and other SignBank databases; verify currency and downstream rights. |

## Fingerspelling

| Source | Direct source | Rights/use note |
| --- | --- | --- |
| FSboard | https://www.kaggle.com/datasets/googleai/fsboard | Primary large-scale source; keep signer-disjoint splits and sensitive-string filtering. |
| ChicagoFSWild / ChicagoFSWild+ | https://home.ttic.edu/~klivescu/ChicagoFSWild.htm | Natural in-the-wild fingerspelling. Review source-video permissions and overlap. Code: https://github.com/chevalierNoir/FS-Detection |
| Google ASL Fingerspelling Recognition data | https://www.kaggle.com/competitions/asl-fingerspelling | Landmark/text competition data distinct from FSboard; competition and data rules govern use. |
| ASL STEM Wiki fingerspelling annotations | https://github.com/microsoft/ASL-STEM-Wiki | Domain-specific detection/alignment annotations; dataset license applies. |
| Purdue RVL-SLLL alphabet material | https://engineering.purdue.edu/RVL/Database/ASL/asl-database-front.htm | Controlled fingerspelling/handshape material under a signed agreement. |

## Other sign languages useful for representation pretraining or transfer

These languages are not ASL. Do not merge labels or evaluate them as ASL.

| Source | Direct source | Rights/use note |
| --- | --- | --- |
| RWTH-PHOENIX-Weather 2014 / 2014T | https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/ | German Sign Language weather benchmark; research terms apply. |
| CSL-Daily | https://ustc-slr.github.io/datasets/2021_csl_daily/ | Chinese Sign Language recognition/translation benchmark; review release terms. |
| AUTSL | https://cvml.ankara.edu.tr/datasets/ | Turkish Sign Language isolated-sign dataset; request/download terms apply. |
| BOBSL | https://www.robots.ox.ac.uk/~vgg/data/bobsl/ | Large British Sign Language broadcast corpus; gated and largely research-oriented. |
| BSL Corpus | https://bslcorpusproject.org/ | Linguistic British Sign Language corpus; dataset-specific access terms apply. |
| DGS Corpus / MEINE DGS | https://www.sign-lang.uni-hamburg.de/meinedgs/overview_en.html | German Sign Language corpus; distinguish viewing access from bulk/training rights. |
| Corpus NGT | https://www.ru.nl/corpusngtuk/ | Sign Language of the Netherlands corpus; review its agreement. |
| SignSuisse | https://signsuisse.sgb-fss.ch/ | Swiss sign-language lexical resource; verify terms before automated collection. |
| WMT Sign Language Translation tasks | https://www.wmt-slt.com/ | Multilingual task datasets and instructions; year-specific terms apply. |
| SignNet-1M | https://signnet.chatsign.ai/ | Multilingual sign dataset and benchmarks; review exact release terms and overlap. |
| Open Source Sign Language | https://ossl.project-edi.org/en/ | Crowdsourced multilingual project; check current export/license before use. |
| Sign Language Translator datasets | https://github.com/sign-language-translator/sign-language-datasets | Smaller multilingual dictionary, sentence and landmark assets; repository declares CC BY 4.0 for its package. |

## Discovery indexes

These indexes are leads, not substitutes for the original source or license.

- Sign Language Dataset Compendium: https://www.sign-lang.uni-hamburg.de/lr/compendium/signlang_dataset_compendium.pdf
- Sign Language Processing TFDS loaders: https://github.com/sign-language-processing/datasets
- SignLanguage Dataset Hub: https://github.com/rudra496/SignLanguage-Dataset-Hub/blob/main/DATASETS.md
- Awesome Sign Language Datasets: https://github.com/ygao-nlp/Awesome-Sign-Lanugage-Datasets
- USTC dataset index: https://ustc-slr.github.io/datasets/
- Sign Language Translator inventory: https://github.com/sign-language-translator/sign-language-datasets

## Private and future UMI sources

- The existing private FLEURS-ASL/FSboard holdout is evaluation-only. It has no public link and must never enter this workspace.
- Convo supplier material is prospective licensed data, not an acquired corpus. It must arrive with recording, signer, rights, alignment and split metadata.
- Future UMI contributor captures and retired cohort clips require consent, provenance, deduplication and a release decision before training.
- `/Users/sam/.local/share/umi/c5-service-corpus-20260929` is operational evaluation/service material and is expressly excluded from training.
