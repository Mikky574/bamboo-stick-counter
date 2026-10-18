# Third-party components and data

Public availability of this repository does not replace the original licenses
or notices of third-party components. No additional license grant for the user's
application code is introduced by this upload.

| Component | Source | License / location |
| --- | --- | --- |
| GeCo2 | https://github.com/jerpelhan/GECO2 | MIT; vendor/geco2/LICENSE |
| Deformable-DETR operators | Bundled by GeCo2, originally SenseTime | Apache-2.0; vendor/geco2/Deformable-DETR/LICENSE. The copied models/ops adapter retains the original headers. |
| SAM2 components | Bundled by GeCo2, originally Meta | Apache-2.0; vendor/geco2/sam2/LICENSE and related notices |
| FSC-147 annotation files | https://github.com/cvlab-stonybrook/LearningToCountEverything | Original repository MIT notice retained in docs/FSC147_LICENSE.txt and the release annotation archive |
| GeCo2 checkpoint | Author's public geco2-assets checkpoint | Original author/repository attribution, source URL and SHA-256 retained in models/provenance.json. The application does not claim ownership of the pretrained weights. |

The Release annotation archive contains the FSC-147 point annotations, category
list and split file downloaded from the original repository. It does not contain
the full dataset image collection. Get images and consult original data terms
through the upstream project.

The public repository includes the user-supplied bamboo test image and generated
synthetic fixtures. Unrelated manufacturer catalog and wood reference images are
not redistributed; their source links remain in the project documentation.

If using GeCo2 in published research, cite:

Pelhan, J., Lukežič, A., and Kristan, M. (2026). Generalized-Scale Object Counting
with Gradual Query Aggregation. AAAI 2026.

If using FSC-147, cite Ranjan et al., Learning To Count Everything, CVPR 2021.

## CountGD++ automatic counting

The optional local text/pseudo-exemplar backend uses the official CountGD++ code:
https://github.com/niki-amini-naieni/CountGDPlusPlus
Pinned revision: 8489f836d75eb407ae778c56cff1f72cc6f8baae. Repository license: MIT;
the repository also retains Grounding DINO / Deformable DETR third-party notices.
The setup script downloads this checkout without removing upstream license files.
Checkpoint source and local SHA-256 are recorded in models/countgdpp-provenance.json.
Language tokenizer/config: google-bert/bert-base-uncased on Hugging Face.

Windows adaptation uses the author's existing PyTorch deformable-attention fallback.
BERT is constructed from its local config before the full checkpoint loads its weights.
Only the unused feature_map_encoder checkpoint branch is excluded, matching the
official demo architecture; all instantiated model parameters load strictly.

Please cite Amini-Naieni and Zisserman, CountGD++: Generalized Prompting for
Open-World Counting, CVPR 2026. The local workflow does not reproduce the paper's
synthetic exemplars or externally enhanced adaptive crops and claims no paper-level
accuracy on bamboo, cotton swabs, or rice.
# TinyCLIP scene routing

The historical optional CPU scene classifier uses the author's TinyCLIP-ViT-8M-16-Text-3M-YFCC15M safetensors model, pinned to Hugging Face revision a2a8c6eaa2549ad66eb7c31b85022bf58273a26c. The model card marks it MIT. See https://huggingface.co/wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M and https://github.com/microsoft/Cream/tree/main/TinyCLIP. This TinyCLIP fallback was not fine-tuned here and is not included in the required v0.2.0 model bundles.

## v0.2.0 model distribution

The default classifier is now a project-fine-tuned ConvNeXt-Small v2 based on
Torchvision's ImageNet initialization. Torchvision v0.21.0 is BSD-3-Clause;
the original license is preserved in docs/licenses/TORCHVISION-v0.21.0-BSD-3-Clause.txt.
Pretrained initialization provenance is preserved with the model metadata.
Code licensing does not replace the terms of original training data, and this
release makes no new blanket MIT grant for all weights or training photographs.

BERT config/tokenizer files originate from google-bert/bert-base-uncased under
Apache-2.0; the original notice is in docs/licenses/BERT-APACHE-2.0.txt.
The CountGD++ original MIT license is in docs/licenses/COUNTGDPP-MIT.txt;
its checkpoint source and digest remain in the release model provenance.

The two required model archives include these original notices, relative file
layouts, the active v2 profile, and per-file/whole-archive SHA-256. They contain
no training images. Vendor source is restored from its pinned upstream revision
and patched by setup, retaining the upstream third-party notices.

Unverified product photographs, collected web training corpora and local output
logs are retained locally rather than newly redistributed in v0.2.0 source.
Previously committed user/synthetic fixtures retain their existing attribution.
The collection scripts remain available as source references; public access to
an image is not a claim that it may be redistributed as a training dataset.
