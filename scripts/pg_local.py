"""Local PostgreSQL migration, backup, and non-destructive restore helpers."""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import tarfile
from datetime import datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))


def run(*args,stdin=None,stdout=None):
    return subprocess.run(args,check=True,stdin=stdin,stdout=stdout)


def require_repo():
    from backend.pg_store import PgRepository
    dsn=os.environ.get('BOM_DATABASE_URL')
    if not dsn: raise SystemExit('请先设置 BOM_DATABASE_URL')
    return PgRepository(dsn)


def migrate():
    repo=require_repo()
    with repo.connection() as conn:
        row=conn.execute("SELECT id FROM users WHERE username='admin' AND role='admin'").fetchone()
    if not row: raise SystemExit('缺少 admin 账号；先在本地初始化管理员')
    print('migrated:',repo.migrate_legacy(ROOT/'output',row['id']))


def backup(directory):
    directory.mkdir(parents=True,exist_ok=False)
    directory.chmod(0o700)
    dump=directory/'bom_workbench.dump'
    with dump.open('wb') as f:
        run('docker','exec','bom-postgres-local','pg_dump','-U','bom_admin','-d','bom_workbench','-Fc',stdout=f)
    archive=directory/'customer-files.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for name in ('output','参考文件'):
            if (ROOT/name).exists():tar.add(ROOT/name,arcname=name)
    with (directory/'SHA256.txt').open('w') as f:
        for path in (dump,archive):
            f.write(f'{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n')
    print(directory)


def restore_to_new(directory,database):
    if not database.startswith('bom_restore_') or not database.replace('_','').isalnum():
        raise SystemExit('恢复目标数据库名称必须以 bom_restore_ 开头且仅包含字母、数字、下划线')
    dump=directory/'bom_workbench.dump'
    if not dump.is_file():raise SystemExit('缺少 bom_workbench.dump')
    run('docker','exec','bom-postgres-local','createdb','-U','bom_admin',database)
    with dump.open('rb') as f:
        run('docker','exec','-i','bom-postgres-local','pg_restore','-U','bom_admin','-d',database,'--no-owner',stdin=f)
    run('docker','exec','bom-postgres-local','psql','-U','bom_admin','-d',database,'-Atc','SELECT count(*) FROM projects')
    print(f'已恢复到独立数据库 {database}；未切换应用连接。')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('migrate','backup','restore-to-new'))
    parser.add_argument('--directory',type=Path)
    parser.add_argument('--database')
    args=parser.parse_args()
    if args.action=='migrate':migrate()
    elif args.action=='backup':backup(args.directory or ROOT.parent/'dist'/('bom-backup-'+datetime.now().strftime('%Y%m%d-%H%M%S')))
    else:
        if not args.directory or not args.database:parser.error('restore-to-new 需要 --directory 和 --database')
        restore_to_new(args.directory,args.database)
