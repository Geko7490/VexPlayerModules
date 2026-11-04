"""Reproducible conversion. Publish catalog only after real ONNX packages pass checks."""
import hashlib
import json
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path
import numpy as np
import onnxruntime as ort
import soundfile as sf
from jsonschema import validate

ROOT = Path(__file__).resolve().parents[1]
OUT = Path('packages')
OUT.mkdir(exist_ok=True)
SOURCES = [
    ('vex-ctc-tr', 'Turkish CTC', 'tr', 'ceyda/wav2vec2-base-760-turkish', 'b16ea5119c4b46aa9e8f79ca57e3d6aa3a204d0a'),
    ('vex-ctc-en', 'English CTC', 'en', 'facebook/wav2vec2-base-960h', '22aad52d435eb6dbaf354bdad9b0da84ce7d6156'),
]
# Official TorchAudio speech fixture, downloaded only for testing, not redistributed.
url = 'https://download.pytorch.org/torchaudio/tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav'
urllib.request.urlretrieve(url, 'speech.wav')
audio, rate = sf.read('speech.wav', dtype='float32')
assert rate == 16000 and audio.ndim == 1 and 400 <= audio.size <= 192000
reports, entries = [], []
for model_id, name, language, source, revision in SOURCES:
    package = OUT / f'{model_id}-1.zip'
    subprocess.run([sys.executable, str(ROOT / 'tools/export_ctc_model.py'), '--model', source,
                    '--revision', revision, '--language', language, '--output', str(package)], check=True)
    with zipfile.ZipFile(package) as z:
        manifest = json.loads(z.read('manifest.json'))
        graph = z.read('model.onnx')
    manifest.update(modelId=model_id, modelVersion='1')
    validate(manifest, json.loads((ROOT / 'schemas/model-manifest.schema.json').read_text()))
    session = ort.InferenceSession(graph, providers=['CPUExecutionProvider'])
    samples = (audio - audio.mean()) / np.sqrt(audio.var() + 1e-7)
    logits = session.run(['logits'], {'input_values': samples[None, :]})[0]
    assert logits.shape == (1, (audio.size - 400) // 320 + 1, len(manifest['vocabulary']))
    assert np.isfinite(logits).all()
    report = {'id': model_id, 'shapeChecks': True, 'realSpeechInference': True,
              'audioSource': url, 'audioSamples': int(audio.size), 'frames': int(logits.shape[1]),
              'androidDeviceVerified': False, 'singingAccuracyVerified': False}
    reports.append(report)
    with zipfile.ZipFile(package, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('model.onnx', graph)
        z.writestr('manifest.json', json.dumps(manifest, ensure_ascii=False))
    entries.append({'id': model_id, 'name': name, 'language': language, 'architecture': 'wav2vec2-character-ctc',
                    'version': '1', 'sizeBytes': package.stat().st_size,
                    'downloadUrl': f'https://github.com/Geko7490/VexPlayerModules/releases/download/ctc-v1/{package.name}',
                    'sha256': hashlib.sha256(package.read_bytes()).hexdigest(), 'minAppVersionCode': 4,
                    'runtime': 'onnxruntime-android-1.23.2', 'license': 'Apache-2.0',
                    'sourceUrl': f'https://huggingface.co/{source}/tree/{revision}',
                    'validation': 'onnx-cpu-real-speech; android-device-and-singing-pending'})
    # Free the inference session before converting/loading the next large model.
    del session, logits, graph
(OUT / 'catalog.json').write_text(json.dumps({'schema': 1, 'models': entries}, indent=2) + '\n')
(OUT / 'validation.json').write_text(json.dumps(reports, indent=2) + '\n')
(OUT / 'MODEL_NOTICE.md').write_bytes((ROOT / 'docs/MODEL_NOTICE.md').read_bytes())
(OUT / 'LICENSE-Apache-2.0.txt').write_bytes((ROOT / 'docs/LICENSE-Apache-2.0.txt').read_bytes())
