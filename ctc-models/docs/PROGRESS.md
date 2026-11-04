# CTC distribution progress — 4 November 2026

Stage 1 complete: Android contract inspected (ONNX character CTC, mono16k, stride320,
receptive400, strict two-file ZIP). Existing manual importer runs native graph inference.
Stage 2/3: pinned Apache-2.0 Turkish/English sources selected, exporter copied from
the app, real-speech CPU smoke checks and immutable release/catalog publication prepared.
Publication completed successfully in Actions run 41894858098:
https://github.com/Geko7490/VexPlayerModules/actions/runs/41894858098
Actual immutable release: https://github.com/Geko7490/VexPlayerModules/releases/tag/ctc-v1
Both public download URLs and sizes passed HEAD checks before catalog commit ba0dc20.
Catalog entries contain SHA-256 of the exact generated packages. Turkish: 90,129,006
bytes; English: 85,992,579 bytes. Real-speech CPU outputs were finite and correctly
shaped for both packages. Original models are Apache-2.0 per linked model cards;
the full license and attribution/modification notices accompany the release.

The publisher is now manual-only: verification-only changes must never attempt to
overwrite an immutable ctc-v1 release. New release versions require updating versioned
manifest/catalog/package names together and a new release tag.

Application model manager/downloader and Android instrumentation are being integrated
on VexPlayer's feature/ai-karaoke-alignment branch. Android and physical-device singing
QA remain pending; do not mark these packages Android-device verified yet.
Four catalog regression tests and a lightweight public-URL verification workflow added.
Minimum app version corrected to versionCode 4 (VexPlayer 1.1.1): older versionCode 3
manual importers reject modelId/modelVersion manifest fields. Package bytes/SHA remain
immutable; public catalog metadata now accurately requires the model-manager release.
