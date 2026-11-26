CREATE TABLE IF NOT EXISTS users (
 id uuid PRIMARY KEY, username text NOT NULL UNIQUE, password_hash text NOT NULL,
 role text NOT NULL CHECK (role IN ('admin','engineer')),
 active boolean NOT NULL DEFAULT true, must_change_password boolean NOT NULL DEFAULT true,
 created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS user_sessions (
 id uuid PRIMARY KEY, user_id uuid NOT NULL REFERENCES users(id), token_hash text NOT NULL UNIQUE,
 csrf_hash text NOT NULL, expires_at timestamptz NOT NULL, revoked_at timestamptz,
 created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS projects (
 id text PRIMARY KEY, owner_id uuid NOT NULL REFERENCES users(id), name text NOT NULL,
 current_drawing_version integer NOT NULL DEFAULT 1, data_version bigint NOT NULL DEFAULT 1,
 state text NOT NULL DEFAULT 'active' CHECK (state IN ('active','trash','archived')),
 snapshot jsonb NOT NULL, source_path text, created_at timestamptz NOT NULL DEFAULT now(),
 updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS drawing_versions (
 project_id text NOT NULL REFERENCES projects(id) ON DELETE CASCADE, version integer NOT NULL,
 sha256 text, page_count integer, workflow_mode text, raw_snapshot jsonb,
 created_at timestamptz NOT NULL DEFAULT now(), PRIMARY KEY(project_id,version)
);
CREATE TABLE IF NOT EXISTS project_files (
 id uuid PRIMARY KEY, project_id text NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
 drawing_version integer, purpose text NOT NULL, storage_key text NOT NULL,
 original_name text NOT NULL, mime text, byte_size bigint, sha256 text,
 export_data_version bigint, state text NOT NULL DEFAULT 'ready',
 created_at timestamptz NOT NULL DEFAULT now(),
 FOREIGN KEY(project_id,drawing_version) REFERENCES drawing_versions(project_id,version)
);
CREATE TABLE IF NOT EXISTS bom_items (
 project_id text NOT NULL REFERENCES projects(id) ON DELETE CASCADE, id text NOT NULL,
 drawing_version integer, collection text NOT NULL, seq_no integer NOT NULL, tag text,
 name text, model text, brand text, quantity numeric, unit text, procurement_scope text,
 source_type text, review_state jsonb, row_version bigint NOT NULL DEFAULT 1, payload jsonb NOT NULL,
 PRIMARY KEY(project_id,id,collection),
 FOREIGN KEY(project_id,drawing_version) REFERENCES drawing_versions(project_id,version),
 CHECK (quantity IS NULL OR quantity >= 0 OR COALESCE((payload->>'is_quantity_adjustment')::boolean,false))
);
ALTER TABLE bom_items DROP CONSTRAINT IF EXISTS bom_items_check;
ALTER TABLE bom_items DROP CONSTRAINT IF EXISTS bom_items_quantity_check;
ALTER TABLE bom_items ADD CONSTRAINT bom_items_quantity_check CHECK (quantity IS NULL OR quantity >= 0 OR COALESCE((payload->>'is_quantity_adjustment')::boolean,false));
CREATE TABLE IF NOT EXISTS bom_evidence (
 id uuid PRIMARY KEY, project_id text NOT NULL, item_id text NOT NULL, item_collection text NOT NULL,
 drawing_version integer, page integer, bbox jsonb, norm_bbox jsonb, raw_text text,
 verified boolean NOT NULL DEFAULT false, payload jsonb,
 FOREIGN KEY(project_id,item_id,item_collection) REFERENCES bom_items(project_id,id,collection) ON DELETE CASCADE,
 FOREIGN KEY(project_id,drawing_version) REFERENCES drawing_versions(project_id,version)
);
CREATE TABLE IF NOT EXISTS review_items (
 project_id text NOT NULL REFERENCES projects(id) ON DELETE CASCADE, id text NOT NULL,
 item_id text, item_collection text, issue_type text NOT NULL, severity text,
 status text NOT NULL, proposed_data jsonb, evidence jsonb, payload jsonb NOT NULL,
 handled_by uuid REFERENCES users(id), handled_at timestamptz,
 PRIMARY KEY(project_id,id),
 FOREIGN KEY(project_id,item_id,item_collection) REFERENCES bom_items(project_id,id,collection)
);
CREATE TABLE IF NOT EXISTS project_revisions (
 project_id text NOT NULL REFERENCES projects(id) ON DELETE CASCADE, version bigint NOT NULL,
 actor_id uuid REFERENCES users(id), source text NOT NULL, summary text NOT NULL,
 snapshot jsonb NOT NULL, created_at timestamptz NOT NULL DEFAULT now(),
 PRIMARY KEY(project_id,version)
);
CREATE TABLE IF NOT EXISTS ai_jobs (
 id text PRIMARY KEY, project_id text NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
 input_data_version bigint NOT NULL, drawing_version integer NOT NULL,
 requested_by uuid REFERENCES users(id), status text NOT NULL, progress jsonb NOT NULL,
 cancel_requested boolean NOT NULL DEFAULT false, heartbeat_at timestamptz,
 created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS ai_runs (
 id uuid PRIMARY KEY, job_id text NOT NULL REFERENCES ai_jobs(id) ON DELETE CASCADE,
 project_id text NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
 input_fingerprint text, drawing_version integer, input_data_version bigint,
 model text, model_config jsonb, prompt_version text, rules_version text,
 memory_versions jsonb, result_validation jsonb, usage jsonb, duration_ms bigint,
 error text, created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS ai_memories (
 id uuid PRIMARY KEY, project_id text NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
 kind text NOT NULL, content jsonb NOT NULL, applicability jsonb,
 source_review_id text, confirmed_by uuid NOT NULL REFERENCES users(id),
 confirmed_data_version bigint NOT NULL, version bigint NOT NULL DEFAULT 1,
 status text NOT NULL DEFAULT 'active', invalidation jsonb,
 created_at timestamptz NOT NULL DEFAULT now(),
 FOREIGN KEY(project_id,source_review_id) REFERENCES review_items(project_id,id) ON DELETE SET NULL (source_review_id)
);
ALTER TABLE ai_memories DROP CONSTRAINT IF EXISTS ai_memories_project_id_source_review_id_fkey;
ALTER TABLE ai_memories ADD CONSTRAINT ai_memories_project_id_source_review_id_fkey FOREIGN KEY(project_id,source_review_id) REFERENCES review_items(project_id,id) ON DELETE SET NULL (source_review_id);
CREATE TABLE IF NOT EXISTS audit_events (
 id bigserial PRIMARY KEY, actor_id uuid REFERENCES users(id), project_id text REFERENCES projects(id),
 action text NOT NULL, request_id text, summary text NOT NULL,
 created_at timestamptz NOT NULL DEFAULT now()
);
CREATE OR REPLACE FUNCTION bom_audit_append_only() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
 IF NOT EXISTS(SELECT 1 FROM pg_roles WHERE rolname=current_user AND rolsuper) THEN
  RAISE EXCEPTION 'audit_events is append-only';
 END IF;
 RETURN OLD;
END $$;
DROP TRIGGER IF EXISTS audit_append_only ON audit_events;
CREATE TRIGGER audit_append_only BEFORE UPDATE OR DELETE ON audit_events FOR EACH ROW EXECUTE FUNCTION bom_audit_append_only();
CREATE OR REPLACE FUNCTION bom_is_admin() RETURNS boolean LANGUAGE sql STABLE SECURITY DEFINER
 SET search_path = public AS $$ SELECT EXISTS(SELECT 1 FROM users WHERE id = nullif(current_setting('app.user_id',true),'')::uuid AND role='admin' AND active) $$;
CREATE OR REPLACE FUNCTION bom_can_access(pid text) RETURNS boolean LANGUAGE sql STABLE SECURITY DEFINER
 SET search_path = public AS $$ SELECT EXISTS(SELECT 1 FROM projects WHERE id=pid AND (owner_id=nullif(current_setting('app.user_id',true),'')::uuid OR bom_is_admin())) $$;
ALTER TABLE projects ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS projects_access ON projects;
CREATE POLICY projects_access ON projects USING (owner_id=nullif(current_setting('app.user_id',true),'')::uuid OR bom_is_admin()) WITH CHECK (owner_id=nullif(current_setting('app.user_id',true),'')::uuid OR bom_is_admin());
DO $$ DECLARE tab text; BEGIN
 FOR tab IN SELECT unnest(ARRAY['drawing_versions','project_files','bom_items','bom_evidence','review_items','project_revisions','ai_jobs','ai_runs','ai_memories']) LOOP
  EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY',tab);
  EXECUTE format('DROP POLICY IF EXISTS project_access ON %I',tab);
  EXECUTE format('CREATE POLICY project_access ON %I USING (project_id IS NULL OR bom_can_access(project_id)) WITH CHECK (project_id IS NULL OR bom_can_access(project_id))',tab);
 END LOOP;
END $$;
ALTER TABLE audit_events ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS project_access ON audit_events;
CREATE POLICY project_access ON audit_events USING (
 (project_id IS NOT NULL AND bom_can_access(project_id)) OR
 (project_id IS NULL AND (actor_id=nullif(current_setting('app.user_id',true),'')::uuid OR bom_is_admin()))
) WITH CHECK (
 (project_id IS NOT NULL AND bom_can_access(project_id)) OR
 (project_id IS NULL AND (actor_id=nullif(current_setting('app.user_id',true),'')::uuid OR bom_is_admin()))
);
CREATE INDEX IF NOT EXISTS idx_projects_owner ON projects(owner_id,state);
CREATE INDEX IF NOT EXISTS idx_ai_jobs_project ON ai_jobs(project_id,status);
CREATE INDEX IF NOT EXISTS idx_audit_project ON audit_events(project_id,created_at);
DO $$ BEGIN
 IF EXISTS(SELECT 1 FROM pg_roles WHERE rolname='bom_app') THEN
  REVOKE UPDATE,DELETE ON audit_events FROM bom_app;
 END IF;
END $$;
