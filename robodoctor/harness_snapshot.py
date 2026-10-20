"""Validated, mode-specific prompt snapshots shared by the RSI tail."""
import hashlib
import json
from pathlib import Path

PROMPT_FILES = ('SKILL.md', 'context/teacher_context.md')
RSI_METHOD = 'rsi_policy_doctor'


def prompt_hashes(root):
    root = Path(root)
    hashes = {}
    for name in PROMPT_FILES:
        path = root / name
        if root.is_symlink() or path.is_symlink() or path.parent.is_symlink():
            raise ValueError(f'Harness prompt must not be a symlink: {name}')
        if not path.is_file() or path.stat().st_size > 1024 * 1024:
            raise ValueError(f'Missing or oversized Harness prompt: {name}')
        data = path.read_bytes()
        if not data.decode('utf-8').strip():
            raise ValueError(f'Empty Harness prompt: {name}')
        hashes[name] = hashlib.sha256(data).hexdigest()
    return hashes


def validate_harness(root, method=RSI_METHOD):
    if method != RSI_METHOD:
        raise ValueError('--harness-dir requires rsi_policy_doctor; direct uses its own prompts')
    manifest = json.loads((Path(root) / 'manifest.json').read_text(encoding='utf-8'))
    hashes = prompt_hashes(root)
    if (not isinstance(manifest, dict) or manifest.get('status') != 'completed'
            or manifest.get('evaluation_method') != RSI_METHOD
            or manifest.get('files') != hashes):
        raise ValueError('Harness snapshot is incomplete, changed, or belongs to another control mode')
    return hashes
