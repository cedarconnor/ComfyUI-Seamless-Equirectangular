"""Saved ERP graphs must resolve without the companion node pack."""
import json
from pathlib import Path

from seamless_equirect.nodes import NODE_CLASS_MAPPINGS


def test_integrated_nodes_registered_without_companion_aliases():
    expected = {
        'EquirectCircularUpscale', 'EquirectCircularVAEDecodeLTX',
        'EquirectCircularVAEDecodeFlux', 'EquirectERPGuider',
        'EquirectSphericalRoPEFlux', 'EquirectLatentDiagnostics',
        'EquirectFrameDiagnostics',
    }
    assert expected <= NODE_CLASS_MAPPINGS.keys()
    assert not any(name.startswith('SpheRoPE') for name in NODE_CLASS_MAPPINGS)


def test_saved_reference_graphs_use_this_pack():
    root = Path(__file__).resolve().parents[1] / 'example_workflows'
    for folder in ['native_reference_20261018', 'seam_diagnosis_20261018']:
        for path in (root / folder).glob('*.json'):
            graph = json.loads(path.read_text())
            for node in graph.values():
                name = node['class_type']
                assert not name.startswith('SpheRoPE'), (path, name)
                if name.startswith('Equirect'):
                    assert name in NODE_CLASS_MAPPINGS, (path, name)
