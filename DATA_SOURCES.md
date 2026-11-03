# Known sign-language data sources

This catalog describes public datasets, acquisition leads and source-specific training conditions. Inclusion is not blanket rights clearance. Verify the exact release, accepted access conditions, attribution and redistribution terms before use.

The owner reports legal approval for public-model-track training with How2Sign and ASL STEM Wiki, treating miner emissions as grants for open-source model work. Models trained with permitted NonCommercial data or inherited NC checkpoints must declare that provenance, use CC BY-NC-SA 4.0 and complete model-bound NC consent. Models with permissive-compatible provenance retain their original terms. Access agreements, confidentiality and source-media restrictions continue to apply.

C6 and later models and their derivatives are public-model-track only. Distribute permitted model weights, not source videos or restricted dataset artifacts. Follow the current rights-review PDF and authenticated submission terms where they provide more specific conditions.


## Translation, fingerspelling and language pretraining

| Source | Direct source | Current handling |
| --- | --- | --- |
| FLEURS-ASL | https://www.kaggle.com/datasets/googleai/fleurs-asl | ASL-to-English recordings and references. Published attribution declares CC BY-SA 4.0. Preserve attribution and publisher splits. |
| 2M-Flores-ASL | https://huggingface.co/datasets/facebook/2M-Flores-ASL | ASL comprehension/translation source; published attribution declares CC BY-SA 4.0. Preserve publisher splits and deduplicate overlapping recordings and text before training. Individual video downloads avoid fetching the full archive. |
| FSboard | https://www.kaggle.com/datasets/googleai/fsboard | Large smartphone fingerspelling corpus; named release components declare CC BY 4.0. Preserve attribution, publisher splits and sensitive-string filtering. The daun_v3 Arrow landmarks have verified coordinate-layout defects and are quarantined for new training. Use original videos or a coordinate-verified alternative. |
| Taskmaster-1 | https://github.com/google-research-datasets/Taskmaster | English conversation text used only for decoder-side language pretraining; CC BY 4.0 in the published attribution. It contains no signing video. |

## ASL sentence translation and continuous signing

| Source | Direct source | Rights/use note |
| --- | --- | --- |
| How2Sign | https://how2sign.github.io/ | Continuous ASL with English, depth and pose. CC BY-NC 4.0 and a research-purpose notice. Approved for public-model-track training under the owner-reported legal review; apply the NC model-license path. Preserve attribution, publisher splits and license records. Do not redistribute source videos or transformed data with model weights. Verify manually realigned timestamps and actual clip durations. |
| OpenASL | https://github.com/chevalierNoir/OpenASL | Open-domain ASL-English video metadata and preparation tools under CC BY-NC-ND 4.0. Approved for public-model-track training under the owner-reported legal review; apply the NC model-license path. Preserve attribution and publisher splits. Source-video rights and availability also apply. Do not redistribute downloaded video or modified dataset artifacts. |
| YouTube-ASL | https://github.com/google-research/google-research/tree/master/youtube_asl | Roughly 1,000 hours of weakly aligned ASL-English. Release is primarily video IDs; source-video availability and platform rights still apply. |
| YouTube-SL-25 | https://arxiv.org/abs/2407.11144 | More than 3,000 hours across 25+ sign languages. Treat the release as a research lead and review current terms before use. |
| ASL STEM Wiki | https://www.microsoft.com/en-us/research/project/asl-stem-wiki/ | Sentence-aligned STEM articles. Publisher reports 64,266 videos, 316 hours and 37 interpreters. The Microsoft Research dataset license limits use to noncommercial, non-revenue-generating research and bars distributing data or modifications. Approved for public-model-track training under the owner-reported legal review; apply the NC model-license path and preserve source-specific conditions. Do not bundle media or modified data with model weights. Code: https://github.com/microsoft/ASL-STEM-Wiki |
| DailyMoth-70h | https://github.com/facebookresearch/ssvp_slt | The publisher releases 48,386 segmented ASL clips and manifests under CC BY-NC 4.0 for self-supervised video pretraining. Its paper describes the pretraining videos as unannotated, so do not count the 70 hours as directly supervised ASL-English translation pairs. Follow the repository's acquisition and attribution instructions. |
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
| ASL Citizen | https://www.microsoft.com/en-us/research/project/asl-citizen/ | Microsoft Research isolated-sign corpus, about 84,000 videos, 52 signers and 2,700 signs, with publisher splits. Its dataset license includes noncommercial/non-revenue research, confidentiality and data-redistribution conditions. Preserve those exact terms; inclusion here is not independent clearance. https://www.microsoft.com/en-us/research/project/asl-citizen/dataset-license/ |
| WLASL | https://github.com/dxli94/WLASL | In-the-wild isolated ASL metadata and clips. C-UDA 1.0 and the publisher README apply; the README limits use to academic/computational research and prohibits commercial usage. Preserve attribution and source-video conditions. Do not redistribute source videos. |
| MS-ASL | https://www.microsoft.com/en-us/download/details.aspx?id=100121 | Large isolated-sign benchmark. Microsoft's download terms govern use. |
| ASLLVD / Boston ASL Lexicon Video Dataset | https://www.bu.edu/asllrp/av/dai-asllvd.html | Publisher documents sign start/end frames and both hands' start/end handshapes. Its [annotation spreadsheet](https://www.bu.edu/asllrp/dai-asllvd-BU_glossing_with_variations_HS_information-extended-urls-RU.xlsx) links videos and session/scene/frame positions. Useful for supervised temporal alignment. Individual processed clips have up to50 buffer frames at each end; some playback is15fps for60fps capture. Verify each file's clock before mapping labels. Compound spreadsheet spans cover the whole compound. Research/education terms prohibit redistribution without permission and restrict commercial use. Resolve overlap with ASLLRP/SignStream. |
| ASL-LEX 2.0 | https://asl-lex.org/ | 2,723 signs with lexical and phonological properties and reference videos. Data/supplement: https://osf.io/zpha4/ |
| Sem-Lex | https://github.com/leekezar/SemLex | 91k+ videos with semantic/phonological tasks. Video access requires separate terms. |
| PopSign RGB v1.0 | https://signdata.cc.gatech.edu/view/datasets/popsign_v1_0/index.html | Publisher card declares CC BY 4.0: 250 signs, 47 signers and 200,686 videos, about 1.1 TB. Preserve Georgia Tech/DPAN attribution and game/variant quality labels. Intended for isolated recognition; the card lists continuous recognition/translation as unsuitable uses. |
| PopSign v2.1 | https://signdata.cc.gatech.edu/view/datasets/popsign_v2_1/index.html | Separate version-specific card declares CC BY 4.0 and 562 signs. It also repeats v1 totals and a current-version 1.0 field; verify the release inventory and overlap before combining versions. |
| Google Isolated Sign Language Recognition / Kaggle ASL Signs | https://www.kaggle.com/competitions/asl-signs/data | Landmark-data release distinct from the fingerspelling competition. Verify the current dataset card, exact release terms and accepted access conditions before acquisition. Check overlap with PopSign before treating it as independent training data. |
| Purdue RVL-SLLL ASL Database | https://engineering.purdue.edu/RVL/Database/ASL/asl-database-front.htm | Handshapes, alphabet, numbers, movement, signs and narratives. Signed license required. |
| ASL Signbank | https://aslsignbank.haskins.yale.edu/ | Lexical database. Check citation and redistribution rules before bulk use. |
| Global Signbank | https://signbank.cls.ru.nl/ | Multiple sign-language lexical databases; dataset-specific terms apply. |
| SignBank download center | https://www.signbank.org/downloads/ | ASL and other SignBank databases; verify currency and downstream rights. |

## Fingerspelling

| Source | Direct source | Rights/use note |
| --- | --- | --- |
| FSboard | https://www.kaggle.com/datasets/googleai/fsboard | Large smartphone fingerspelling corpus; named release components declare CC BY 4.0. Preserve attribution, publisher splits and sensitive-string filtering. The daun_v3 Arrow landmarks have verified coordinate-layout defects and are quarantined for new training. Use original videos or a coordinate-verified alternative. |
| ChicagoFSWild / ChicagoFSWild+ | https://home.ttic.edu/~klivescu/ChicagoFSWild.htm | Publisher reports 7,304 labeled sequences and 160 signers for ChicagoFSWild, and 55,232 sequences and 260 signers for ChicagoFSWild+. The owner approved training on the train split of both releases, with provenance disclosure; never train on their dev/test splits. The publisher page has no explicit reuse-license text, so retain the owner decision and originating-video conditions. Deduplicate repeated intervals and resolve conflicting labels. Code: https://github.com/chevalierNoir/FS-Detection |
| Google ASL Fingerspelling Recognition data | https://www.kaggle.com/competitions/asl-fingerspelling/data | Landmark/text competition data distinct from FSboard. The publisher's data page lists CC BY 4.0 and says access requires agreeing to the competition rules after sign-in. Confirm the accepted rules and attribution before training or releasing derivative weights. The pilot [OpenGlasses CTC checkpoint](https://huggingface.co/Skunk0/openglasses-fingerspelling-ctc) declares Apache-2.0 for its weights and conversion and says it was trained on this corpus; those declarations do not independently establish source compliance. |
| ASL STEM Wiki fingerspelling annotations | https://github.com/microsoft/ASL-STEM-Wiki | Publisher supplies fingerspelling frame spans and spelled text. Verify native clocks, span/text counts and label types before using annotations as training targets; a guessed frame rate does not establish timing provenance. Dataset terms and the NC model-license path apply. Retain source metadata and derivatives privately. |
| Purdue RVL-SLLL alphabet material | https://engineering.purdue.edu/RVL/Database/ASL/asl-database-front.htm | Controlled fingerspelling/handshape material under a signed agreement. |

## Other sign languages useful for representation pretraining or transfer

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

- Sign Language Dataset Compendium: https://www.sign-lang.uni-hamburg.de/lr/compendium/signlang_dataset_compendium.pdf
- Sign Language Processing TFDS loaders: https://github.com/sign-language-processing/datasets
- SignLanguage Dataset Hub: https://github.com/rudra496/SignLanguage-Dataset-Hub/blob/main/DATASETS.md
- Awesome Sign Language Datasets: https://github.com/ygao-nlp/Awesome-Sign-Lanugage-Datasets
- USTC dataset index: https://ustc-slr.github.io/datasets/
- Sign Language Translator inventory: https://github.com/sign-language-translator/sign-language-datasets
