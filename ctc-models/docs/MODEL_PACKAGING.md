# Reproduce the model release

Pinned model sources, revisions and dependencies are in tools/publish_packages.py and
.github/workflows/publish-ctc.yml. `export_ctc_model.py` is the VexPlayer exporter.
It exports dynamic ONNX opset 17 and quantizes MatMul weights only (no ConvInteger).
It tests 1 s, 4.02 s and 12 s input shapes. Publication additionally tests real
speech input, validates the manifest schema and calculates package byte size/SHA-256.

`gh release create` publishes immutable v1 assets without committing model binaries.
Only after publication and public URL checks does the workflow commit catalog.json.
A failed publication leaves the existing catalog unchanged; retry publication with
a NEW release/version after inspecting the failure, never replace immutable assets.

Speech fixture comes from the official TorchAudio forced-alignment tutorial. It is
downloaded for inference testing and is not included in release assets. CPU inference
checks do not prove Android loading, word-boundary accuracy or singing performance.
