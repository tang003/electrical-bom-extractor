FROM --platform=$BUILDPLATFORM mcr.microsoft.com/dotnet/sdk:8.0-bookworm-slim AS cad-build
WORKDIR /build
COPY tools/cad_converter/*.csproj ./
RUN dotnet restore
COPY tools/cad_converter/*.cs ./
RUN dotnet publish -c Release -o /publish --no-restore --self-contained false -p:UseAppHost=false

FROM mcr.microsoft.com/dotnet/runtime:8.0-bookworm-slim AS dotnet-runtime

FROM python:3.12-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DOTNET_ROOT=/usr/share/dotnet \
    DOTNET_CLI_TELEMETRY_OPTOUT=1 \
    PATH="/usr/share/dotnet:${PATH}" \
    HOME=/data \
    ELECTRICAL_BOM_SERVER_MODE=1 \
    ELECTRICAL_BOM_OUTPUT_DIR=/data/projects \
    ELECTRICAL_BOM_VLM_CONFIG=/data/config/vlm_config.json \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates libgl1 libglib2.0-0 libgomp1 libicu72 libssl3 libstdc++6 zlib1g fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 bom && useradd --uid 10001 --gid bom --home-dir /data --no-create-home bom \
    && mkdir -p /data/projects /data/config /data/logs /data/dataset \
    && chown -R bom:bom /data
COPY --from=dotnet-runtime /usr/share/dotnet /usr/share/dotnet
WORKDIR /app
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY backend ./backend
COPY bom_extractor ./bom_extractor
COPY frontend ./frontend
COPY scripts/migrate_server.py ./scripts/migrate_server.py
COPY --from=cad-build /publish ./tools/cad_converter/publish
USER bom:bom
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4).read()"
CMD ["python", "-m", "uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers"]
