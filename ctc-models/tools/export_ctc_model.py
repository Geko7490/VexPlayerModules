"""Export a language-specific wav2vec2 character-CTC model for the Android prototype.

Requires a separate desktop Python environment; these are NOT app dependencies:
  pip install torch==2.9.0 transformers==4.57.1 onnx==1.19.1 onnxruntime==1.23.2
No audio or lyrics are uploaded. Hugging Face weights download only on this explicit command.
"""
import argparse
import json
import tempfile
import zipfile
from pathlib import Path


def main():
    import numpy as np
    import onnxruntime as ort
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoFeatureExtractor, AutoModelForCTC, AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Hugging Face repo ID or local directory")
    parser.add_argument("--revision", required=True, help="Pinned model commit SHA (or local)")
    parser.add_argument("--language", required=True, choices=["tr", "en"])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    kwargs = {} if args.revision == "local" else {"revision": args.revision}
    model = AutoModelForCTC.from_pretrained(args.model, trust_remote_code=False, attn_implementation="eager", **kwargs).eval().cpu()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False, **kwargs)
    extractor = AutoFeatureExtractor.from_pretrained(args.model, trust_remote_code=False, **kwargs)
    if model.config.model_type != "wav2vec2" or list(model.config.conv_stride) != [5, 2, 2, 2, 2, 2, 2] or list(model.config.conv_kernel) != [10, 3, 3, 3, 3, 2, 2]:
        raise ValueError("Only wav2vec2 with 320-sample stride and 400-sample receptive field is supported")
    vocab = tokenizer.get_vocab()
    if len(vocab) > 256 or set(vocab.values()) != set(range(len(vocab))):
        raise ValueError("Contiguous character vocabulary required")
    letters = [k for k in vocab if len(k) == 1 and k.isalpha()]
    if not letters or any(len(k) > 1 and not k.startswith("<") and not k.startswith("[") for k in vocab):
        raise ValueError("A character CTC vocabulary is required")
    manifest = {
        "schema": 1, "name": f"{args.model}@{args.revision}", "language": args.language,
        "vocabulary": vocab, "blankId": model.config.pad_token_id,
        "sampleRate": 16000, "strideSamples": 320, "receptiveSamples": 400,
        "inputName": "input_values", "outputName": "logits",
        "lowercase": sum(c.islower() for c in letters) >= sum(c.isupper() for c in letters),
        "wordDelimiter": tokenizer.word_delimiter_token,
        "normalize": bool(extractor.do_normalize),
    }
    if extractor.sampling_rate != 16000 or manifest["wordDelimiter"] not in vocab:
        raise ValueError("Incompatible frontend")

    class AcousticGraph(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base

        def forward(self, input_values):
            return self.base(input_values, return_dict=False)[0]

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        raw, quantized = root / "raw.onnx", root / "model.onnx"
        with torch.inference_mode():
            torch.onnx.export(AcousticGraph(model), (torch.zeros(1, 16000),), str(raw),
                              input_names=["input_values"], output_names=["logits"],
                              dynamic_axes={"input_values": {1: "samples"}, "logits": {1: "frames"}},
                              opset_version=17, dynamo=False, external_data=False)
        # ConvInteger kernels are not universally supported on Android; quantize MatMul only.
        quantize_dynamic(str(raw), str(quantized), weight_type=QuantType.QInt8,
                         op_types_to_quantize=["MatMul"], use_external_data_format=False)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        session = ort.InferenceSession(str(quantized), options, providers=["CPUExecutionProvider"])
        for n in (16000, 64320, 192000):
            samples = np.zeros((1, n), dtype=np.float32)
            output = session.run(["logits"], {"input_values": samples})[0]
            expected = (1, (n - 400) // 320 + 1, len(vocab))
            if output.shape != expected or not np.isfinite(output).all():
                raise ValueError(f"Output contract mismatch: {output.shape} != {expected}")
        if quantized.stat().st_size > 160 * 1024 * 1024 - 65536:
            raise ValueError("Quantized model exceeds Android prototype's 160 MB package budget")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.write(quantized, "model.onnx")
            archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False))
    print(f"Created {args.output}. Desktop shape checks passed; Android accuracy/performance still require device validation.")


if __name__ == "__main__":
    main()
