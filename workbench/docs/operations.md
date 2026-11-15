# Local operational deployment

The Workbench has two persistence modes:

- `local` (default): immutable run JSON, inputs and artifacts live below
  `PANORAI_WORKBENCH_HOME` or `outputs/workbench`.
- `replicated`: the same local directory remains the execution cache, run
  documents are upserted into PostgreSQL, and content-addressed inputs plus run
  artifacts are mirrored to one fixed S3 bucket.

The browser never sends database URLs, SQL, bucket names, object keys or
credentials. Replicated mode fails closed if any required setting is absent or
if the optional `panorai-workbench[ops]` dependencies are unavailable.

## Start

Copy `.env.example` to `.env`, replace every example secret, and run:

```bash
docker compose up --build
```

The stack publishes only:

- `http://127.0.0.1:5173` — frontend;
- `http://127.0.0.1:8000/api/v1/health/ready` — backend readiness and
  persistence diagnostics.

PostgreSQL and MinIO are reachable only on the private Compose network. Named
volumes retain metadata, objects and the local execution cache across container
restarts. `docker compose down` preserves them; removing volumes is an explicit
destructive operation and is not part of normal shutdown.

## Security boundary

This deployment has no authentication, authorization, tenant isolation or
untrusted-upload sandbox. Host ports are bound to loopback and the ASGI app
accepts only local/container host names. Do not change those bindings or place
the service behind a remotely reachable proxy until an authentication design,
CSRF policy, resource quotas and upload isolation have been implemented and
reviewed.

## Health semantics

`/api/v1/health/live` proves that the process can answer. `/api/v1/health/ready`
also probes PostgreSQL and the fixed S3 bucket in replicated mode and returns
HTTP 503 if either is unavailable. Local mode reports the filesystem backends
without importing PostgreSQL, boto3 or optional scientific runtimes.

The replicated services are durable mirrors in this phase. The filesystem is
still required as the execution cache and read path, so deleting it is not a
supported recovery workflow yet.
