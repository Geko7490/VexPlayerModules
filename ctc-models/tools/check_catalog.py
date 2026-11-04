import json
import urllib.request
from pathlib import Path

catalog = json.loads(Path('ctc-models/catalog.json').read_text())
assert catalog['schema'] == 1 and len(catalog['models']) == 2
for model in catalog['models']:
    with urllib.request.urlopen(urllib.request.Request(model['downloadUrl'], method='HEAD')) as response:
        assert response.status == 200
        assert int(response.headers['Content-Length']) == model['sizeBytes']
print('Both published model URLs and sizes verified')
