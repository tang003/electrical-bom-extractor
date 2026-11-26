"""Run against a local PostgreSQL URL; creates and removes only test records."""
import os
import shutil
import time
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4,UUID

from fastapi.testclient import TestClient
import psycopg
import pymupdf

from backend.auth import ph
from backend.pg_store import Conflict, PgRepository
from bom_extractor.models import ExtractionResult,BOMItem,ReviewItem,PageInfo


def run():
    repo=PgRepository(os.environ['BOM_DATABASE_URL'])
    prefix='smoke_'+uuid4().hex[:10]
    ids=[uuid4() for _ in range(4)]
    project_id='proj_'+prefix
    passwords=['test-password-'+str(i) for i in range(4)]
    try:
        with repo.connection() as conn:
            for i,user_id in enumerate(ids):
                conn.execute('INSERT INTO users(id,username,password_hash,role,must_change_password) VALUES(%s,%s,%s,%s,%s)',(user_id,prefix+str(i),ph.hash(passwords[i]),'admin' if i==0 else 'engineer',i==3))
        test_output=Path(__file__).resolve().parent.parent/'output'/project_id
        test_output.mkdir(parents=True,exist_ok=False)
        source_pdf=test_output/'source.pdf'
        with pymupdf.open() as pdf:
            pdf.new_page(width=400,height=400)
            pdf.save(source_pdf)
        sample=ExtractionResult(project_id,'smoke.pdf','隔离验收临时项目','','SCHEMATIC_MODE',1,
            pages=[PageInfo(1,'CONTROL_CIRCUIT','控制回路',width=400,height=400)],
            itemized_bom=[BOMItem('smoke_item',1,'-K1',tags=['-K1'],name='继电器',quantity=1)])
        assert repo.save_result(sample,ids[1],source='test',summary='隔离验收临时项目',source_path=str(source_pdf))==1
        try:
            with repo.connection(ids[1]) as conn:
                conn.execute('UPDATE audit_events SET summary=summary WHERE project_id=%s',(project_id,))
        except psycopg.errors.InsufficientPrivilege: pass
        else: raise AssertionError('audit log can be modified by app role')
        with repo.connection(ids[2]) as conn:
            assert conn.execute('SELECT id FROM projects WHERE id=%s',(project_id,)).fetchone() is None
            assert conn.execute('SELECT id FROM bom_items WHERE project_id=%s',(project_id,)).fetchone() is None
            assert conn.execute('SELECT id FROM project_files WHERE project_id=%s',(project_id,)).fetchone() is None
        try: repo.save_result(sample,ids[1],expected_version=0)
        except Conflict: pass
        else: raise AssertionError('stale version accepted')

        import backend.app as backend
        with TestClient(backend.app) as web:
            job_id=uuid4().hex
            backend.ai_jobs.save({'id':job_id,'project_id':project_id,'requested_by':str(ids[1]),'input_data_version':1,'drawing_version':1,'status':'queued','cancel_requested':False,'message':'测试重启恢复'})
            def login(i):
                client=TestClient(backend.app)
                response=client.post('/api/auth/login',json={'username':prefix+str(i),'password':passwords[i]})
                assert response.status_code==200,response.text
                assert 'Max-Age=2592000' in response.headers['set-cookie']
                return client
            admin,owner,other,first=(login(i) for i in range(4))
            created=admin.post('/api/auth/users',json={'username':prefix+'_new','password':'12345678','role':'engineer'},headers={'X-BOM-CSRF':admin.cookies['bom_csrf']})
            assert created.status_code==200,(created.status_code,created.text)
            created_id=UUID(created.json()['id']);ids.append(created_id)
            newcomer=TestClient(backend.app)
            assert newcomer.post('/api/auth/login',json={'username':prefix+'_new','password':'12345678'}).status_code==200
            assert newcomer.get('/api/projects').status_code==403
            disabled=admin.post(f'/api/auth/users/{created_id}/active/false',headers={'X-BOM-CSRF':admin.cookies['bom_csrf']})
            assert disabled.status_code==200,(disabled.status_code,disabled.text)
            assert newcomer.get('/api/projects').status_code==401
            assert first.get('/api/projects').status_code==403
            same_password=first.post('/api/auth/password',json={'old_password':passwords[3],'new_password':passwords[3]},headers={'X-BOM-CSRF':first.cookies['bom_csrf']})
            assert same_password.status_code==400
            changed_password=first.post('/api/auth/password',json={'old_password':passwords[3],'new_password':'12345678'},headers={'X-BOM-CSRF':first.cookies['bom_csrf']})
            assert changed_password.status_code==200,(changed_password.status_code,changed_password.text)
            assert first.get('/api/projects').status_code==401
            passwords[3]='12345678'
            first=login(3)
            assert first.get('/api/projects').status_code==200
            assert [p['project_id'] for p in owner.get('/api/projects').json()]==[project_id]
            assert owner.get(f'/api/projects/{project_id}/source').status_code==200
            assert owner.get(f'/api/projects/{project_id}/pages/1/image').status_code==200
            recovered=owner.get(f'/api/projects/{project_id}/ai-jobs/{job_id}')
            assert recovered.status_code==200 and recovered.json()['status']=='interrupted',(recovered.status_code,recovered.text)
            current=owner.get(f'/api/projects/{project_id}').json()
            assert current['data_version']==1
            endpoint=f'/api/projects/{project_id}/bom/item'
            payload={'view_mode':'itemized','id':'smoke_item','name':'测试继电器'}
            assert owner.put(endpoint,json=payload,headers={'X-BOM-CSRF':owner.cookies['bom_csrf']}).status_code==428
            changed=owner.put(endpoint,json=payload,headers={'X-BOM-CSRF':owner.cookies['bom_csrf'],'If-Match':'1'})
            assert changed.status_code==200,(changed.status_code,changed.text)
            assert changed.json()['data_version']==2
            assert changed.headers.get('X-Request-ID')
            with repo.connection(ids[1]) as conn:
                assert conn.execute("SELECT 1 FROM audit_events WHERE project_id=%s AND request_id=%s",(project_id,changed.headers['X-Request-ID'])).fetchone()
            assert owner.put(endpoint,json=payload,headers={'X-BOM-CSRF':owner.cookies['bom_csrf'],'If-Match':'1'}).status_code==409
            assert all(p['project_id']!=project_id for p in other.get('/api/projects').json())
            for method,path in [('GET',f'/api/projects/{project_id}'),('GET',f'/api/projects/{project_id}/source'),('GET',f'/api/projects/{project_id}/pages/1/image'),('GET',f'/api/projects/{project_id}/ai-jobs/{"0"*32}'),('POST',f'/api/projects/{project_id}/ai-extract'),('POST',f'/api/projects/{project_id}/trash')]:
                r=other.request(method,path,headers={'X-BOM-CSRF':other.cookies.get('bom_csrf','')})
                assert r.status_code==404,(path,r.status_code,r.text)
            r=owner.post('/api/projects/clear',headers={'X-BOM-CSRF':owner.cookies['bom_csrf']})
            assert r.status_code==403,(r.status_code,r.text)
            assert admin.get(f'/api/projects/{project_id}').status_code==200
            r=admin.post(f'/api/projects/{project_id}/transfer',json={'owner_id':str(ids[2])},headers={'X-BOM-CSRF':admin.cookies['bom_csrf']})
            assert r.status_code==200,(r.status_code,r.text)
            assert owner.get(f'/api/projects/{project_id}').status_code==404
            assert other.get(f'/api/projects/{project_id}').status_code==200
            r=admin.post(f'/api/auth/users/{ids[1]}/reset',json={'password':'changed-password-123'},headers={'X-BOM-CSRF':admin.cookies['bom_csrf']})
            assert r.status_code==200,(r.status_code,r.text)
            assert owner.get('/api/projects').status_code==401
            r=admin.post(f'/api/projects/{project_id}/trash',headers={'X-BOM-CSRF':admin.cookies['bom_csrf']})
            assert r.status_code==200,(r.status_code,r.text)
            assert project_id in [x['id'] for x in admin.get('/api/projects/trash').json()]
            backend.store.results.pop(project_id,None)
            backend.store.data_versions.pop(project_id,None)
            r=admin.post(f'/api/projects/{project_id}/restore',headers={'X-BOM-CSRF':admin.cookies['bom_csrf']})
            assert r.status_code==200,(r.status_code,r.text)
            assert admin.get(f'/api/projects/{project_id}').status_code==200
            assert len(admin.get(f'/api/projects/{project_id}/revisions').json())>=3
        with repo.connection(ids[0]) as conn:
            row=conn.execute('SELECT snapshot,data_version FROM projects WHERE id=%s',(project_id,)).fetchone()
        result=ExtractionResult.from_dict(row['snapshot'])
        review=ReviewItem('memory_review','INFO','AI_FIELD_CLASSIFICATION','-K1','字段确认','人工审核','采纳',1,
            status='ACCEPTED',proposed_data={'model':'TEST-100'})
        result.review_items=[review]
        version=repo.save_result(result,ids[0],expected_version=row['data_version'])
        with repo.connection(ids[2]) as conn:
            memory=conn.execute('SELECT status,version,source_review_id FROM ai_memories WHERE project_id=%s',(project_id,)).fetchone()
            assert memory=={'status':'active','version':1,'source_review_id':'memory_review'},memory
        from backend.auth import actor
        context_token=actor.set({'id':ids[2]})
        try:
            memories=backend._confirmed_field_memories(project_id,'-K1')
            assert len(memories)==1 and memories[0]['version']==1
            backend._record_field_analysis(project_id,'smoke_item','test-fingerprint',version,memories,'completed',{'model':'TEST-100'},{},True,time.time())
        finally:
            actor.reset(context_token)
        with repo.connection(ids[2]) as conn:
            run=conn.execute("SELECT memory_versions FROM ai_runs WHERE project_id=%s AND input_fingerprint='test-fingerprint'",(project_id,)).fetchone()
            assert len(run['memory_versions'])==1 and run['memory_versions'][0]['version']==1
        with repo.connection(ids[1]) as conn:
            assert conn.execute('SELECT id FROM ai_memories WHERE project_id=%s',(project_id,)).fetchone() is None
        review.status='IGNORED'
        repo.save_result(result,ids[0],expected_version=version)
        with repo.connection(ids[2]) as conn:
            assert conn.execute('SELECT status FROM ai_memories WHERE project_id=%s',(project_id,)).fetchone()['status']=='inactive'
        print('PostgreSQL integration smoke passed')
    finally:
        local_config=Path(__file__).resolve().parent.parent/'.local/postgres.env'
        admin_dsn=os.environ.get('BOM_TEST_ADMIN_DATABASE_URL')
        if not admin_dsn and local_config.is_file():
            cfg=dict(line.split('=',1) for line in local_config.read_text().splitlines())
            admin_dsn=f"postgresql://{cfg['POSTGRES_USER']}:{quote(cfg['POSTGRES_PASSWORD'])}@127.0.0.1:55432/{cfg['POSTGRES_DB']}"
        if not admin_dsn: raise RuntimeError('需要 BOM_TEST_ADMIN_DATABASE_URL 清理临时审计记录')
        with psycopg.connect(admin_dsn) as conn:
            conn.execute('DELETE FROM audit_events WHERE project_id=%s OR actor_id=ANY(%s)',(project_id,ids))
        with repo.connection(ids[0]) as conn:
            conn.execute('DELETE FROM projects WHERE id=%s',(project_id,))
            conn.execute('DELETE FROM user_sessions WHERE user_id=ANY(%s)',(ids,))
            conn.execute('DELETE FROM users WHERE id=ANY(%s)',(ids,))
        test_output=Path(__file__).resolve().parent.parent/'output'/project_id
        if test_output.is_dir() and project_id.startswith('proj_smoke_'):
            shutil.rmtree(test_output)


if __name__=='__main__': run()
