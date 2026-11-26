"""One-time migration from the file-backed server into a separate PostgreSQL database.

Run in a transient container with BOM_DB_ADMIN_URL, BOM_DATABASE_URL and
BOM_APP_PASSWORD set. The running web container never receives the admin URL.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from uuid import uuid4

import psycopg
from psycopg import sql

from backend.pg_store import PgRepository


def migrate(data_dir: Path, expected_projects: int | None = None) -> int:
    admin_url = os.environ['BOM_DB_ADMIN_URL']
    app_url = os.environ['BOM_DATABASE_URL']
    app_password = os.environ['BOM_APP_PASSWORD']
    account = json.loads((data_dir / 'config' / 'account.json').read_text(encoding='utf-8'))
    username, password_hash = account['username'], account['password_hash']
    if username != 'admin' or account.get('schema_version') != 1 or not re.fullmatch(r'pbkdf2_sha256\$\d+\$[0-9a-f]{64}\$[0-9a-f]{64}', password_hash):
        raise ValueError('旧服务器账号格式与预期不符，迁移已停止')
    source_states = list((data_dir / 'projects').glob('*/project_state.json'))
    if expected_projects is not None and len(source_states) != expected_projects:
        raise ValueError(f'预期 {expected_projects} 个项目，实际发现 {len(source_states)} 个')
    with psycopg.connect(admin_url) as conn:
        if not conn.execute("SELECT 1 FROM pg_roles WHERE rolname='bom_app'").fetchone():
            conn.execute(sql.SQL('CREATE ROLE bom_app LOGIN PASSWORD {}').format(sql.Literal(app_password)))
    admin_repo = PgRepository(admin_url)
    admin_repo.migrate_schema()
    with psycopg.connect(admin_url) as conn:
        conn.execute('GRANT USAGE ON SCHEMA public TO bom_app')
        conn.execute('GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO bom_app')
        conn.execute('GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO bom_app')
        conn.execute('REVOKE UPDATE, DELETE ON audit_events FROM bom_app')
        row = conn.execute("SELECT id FROM users WHERE username='admin'").fetchone()
        if row is None:
            admin_id = uuid4()
            conn.execute('INSERT INTO users(id,username,password_hash,role,must_change_password) VALUES(%s,%s,%s,%s,%s)',
                         (admin_id, username, password_hash, 'admin', False))
        else:
            admin_id = row[0]
    migrated = admin_repo.migrate_legacy(data_dir / 'projects', admin_id)
    with psycopg.connect(app_url) as conn:
        count = conn.execute('SELECT count(*) FROM projects').fetchone()[0]
    if count != len(source_states):
        raise RuntimeError(f'数据库项目数 {count} 与旧版项目数 {len(source_states)} 不一致')
    print(f'Migrated {len(migrated)} projects; database now has {count} projects.')
    return count


if __name__ == '__main__':
    expected = os.environ.get('BOM_EXPECT_PROJECTS')
    migrate(Path(os.environ.get('ELECTRICAL_BOM_DATA_DIR', '/data')),
            int(expected) if expected else None)
