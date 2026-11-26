"""Production transition keeps the previous login and rejects Linux file secrets."""

import base64
import hashlib
import json

import pytest

from backend.auth import ph, verify_password_hash
from bom_extractor.credentials import _read_key, _write_key


def test_legacy_password_hash_is_accepted_without_weakening_argon2():
    salt = bytes(range(32))
    digest = hashlib.pbkdf2_hmac('sha256', b'correct-password', salt, 600_000)
    old_hash = f'pbkdf2_sha256$600000${salt.hex()}${digest.hex()}'
    assert verify_password_hash(old_hash, 'correct-password')
    assert not verify_password_hash(old_hash, 'wrong-password')
    assert not verify_password_hash(old_hash.replace('$600000$', '$20$'), 'correct-password')
    assert verify_password_hash(ph.hash('new-password'), 'new-password')
    assert not verify_password_hash('invalid-hash', 'new-password')


def test_server_mode_rejects_unencrypted_file_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv('ELECTRICAL_BOM_SERVER_MODE', '1')
    path = tmp_path / 'credentials.json'
    path.write_text(json.dumps({'storage': 'user-file', 'value': base64.b64encode(b'example-key').decode()}), encoding='utf-8')
    assert _read_key(path) == ''
    with pytest.raises(ValueError, match='生产服务密钥'):
        _write_key(path, 'new-key')
