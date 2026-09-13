# Licenses and inference provenance

This is a standalone inference assembly. It is not an official upstream release
or a claim of measured translation accuracy. Required public author attribution
is retained; no private training inventory is supplied.

| Component | Public source | Declared terms |
|---|---|---|
| SHuBERT translation and fine-tuned hand/face tensors | https://huggingface.co/ShesterG/SHuBERT | Publisher model repository declares MIT; revision 578a0233e770c8ce4dc75d859b91fdea7c34f5aa |
| SHuBERT demo architecture and preprocessing | https://huggingface.co/spaces/ShesterG/TTIC-SHuBERT-ASLVideo-to-EnglishText/tree/69d3d77aa4a4fec89048f5417d44fa717e36f6f8 | Publisher Space declares MIT |
| Fairseq library | https://github.com/facebookresearch/fairseq | MIT; included notice in licenses/fairseq-MIT.txt |
| DINOv2 architecture | https://github.com/facebookresearch/dinov2/tree/7764ea0f912e53c92e82eb78a2a1631e92725fc8 | Apache-2.0; license retained with source |
| ByT5 tokenizer/base architecture lineage | https://huggingface.co/google/byt5-base | Apache-2.0 |
| MediaPipe face/hand task assets and runtime | https://github.com/google-ai-edge/mediapipe | Runtime: Apache-2.0. Task assets are distributed in the SHuBERT model repository under its declared MIT terms; no broader asset-rights claim is made |
| Standalone wrapper and compatibility changes | Included source | Apache-2.0, see LICENSE |

SHuBERT citation: *SHuBERT: Self-Supervised Sign Language Representation Learning
via Multi-Stream Cluster Prediction*, https://arxiv.org/abs/2411.16765.
ByT5 citation: *ByT5: Towards a token-free future with pre-trained byte-to-byte
models*, https://arxiv.org/abs/2105.13626.
DINOv2 citation: *DINOv2: Learning Robust Visual Features without Supervision*,
https://arxiv.org/abs/2304.07193.

The publisher model card is sparse: MIT is its declared license, not a detailed
independent audit of upstream training-data rights. This package grants no rights
to upstream source videos or annotations, which are not included. Retain all
applicable upstream attribution when redistributing. Python dependency licenses
remain with their separately installed distributions.

Changes in this assembly: package-relative paths; standalone serial CLI; explicit
CPU/MPS device selection; offline model loading; OpenCV video-reader compatibility;
five-beam decoder configuration; removal of example entry points and diagnostic
source-path prints; inference-only DINO architecture inventory; tensor-only
safetensors serialization with exact tensor-value checks; removal of training
metadata from configuration. No model tensor was trained, pruned or quantized.

The SHA-256 inventory binds the exact distributed source, model and configuration.
No optimizer, training arguments, RNG state, corpus files or experiment records
are part of this distribution.
