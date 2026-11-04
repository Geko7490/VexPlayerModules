# Vex CTC model format v1

The ZIP contains exactly `model.onnx` and `manifest.json`; directories, duplicates,
external tensors and arbitrary paths are rejected. Uncompressed limit: 160 MiB;
manifest: 64 KiB. Schema: ../schemas/model-manifest.schema.json.

Runtime: ONNX Runtime Android 1.23.2, CPU, 2 intra-op / 1 inter-op threads.
Input: `input_values`, float32 [1,N], mono 16 kHz, mean/variance normalization
with epsilon 1e-7. Output: `logits`, float32 [1,T,V], T=floor((N-400)/320)+1.
Vocab IDs must be contiguous; blankId must identify the CTC blank, delimiter `|`.
Only character tokenization is supported. Original word text is retained separately.
Times have 20 ms acoustic resolution, stored as integer milliseconds.

The Android installer validates the vocabulary and runs the actual graph before
switching an atomic active-generation pointer. Translation `.task` models are unrelated.
Installation state is local device state and is never claimed by the shared catalog.
