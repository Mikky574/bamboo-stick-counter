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
