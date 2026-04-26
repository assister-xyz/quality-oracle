# QO-059 L3 activation harness — full Claude Code runtime in a sandboxed image.
#
# Why Ubuntu 24.04 + Python 3.13 + Node 20 (npm)?
#   - claude-agent-sdk requires Python 3.10+ and is pip-distributed.
#   - @anthropic-ai/claude-code is npm-distributed and embeds the CLI.
#   - Both are loaded by `runner.py` (single-process runtime).
#
# Egress: this image has NO direct bridge access. It runs with
# `--network=container:laureum-proxy`, sharing namespace with the squid sidecar
# defined in ../laureum-proxy/. AC2.
#
# Skill mount: callers `-v <skill-dir>:/eval/.claude/skills/<name>:ro` so the
# SDK discovers it via setting_sources=["project"]. AC1, AC9.
#
# Session state: callers `--tmpfs /root/.claude:rw,size=64m` so the SDK can
# persist session files inside an in-memory writable layer. AC9.
#
# No `--bare` CLI flag — the SDK is configured programmatically from runner.py
# via ClaudeAgentOptions(setting_sources=["project"]). N4.

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Python 3.13 from deadsnakes (Ubuntu 24.04 ships 3.12 — we want 3.13 for
# parity with the host quality-oracle service).
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates curl gnupg jq \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.13 python3.13-venv python3-pip \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# claude-code CLI (npm) — pinned in package.json for reproducibility (AC5).
COPY package.json /opt/runner/package.json
RUN npm install -g @anthropic-ai/claude-code && which claude

# claude-agent-sdk + anthropic (pip).
# `--ignore-installed` because Ubuntu's python3-pip pulls in distro-managed
# PyJWT/etc.; we want our pinned wheels regardless.
RUN python3.13 -m pip install --break-system-packages --ignore-installed \
        claude-agent-sdk \
        anthropic

WORKDIR /eval
COPY runner.py /usr/local/bin/laureum-runner
RUN chmod +x /usr/local/bin/laureum-runner

# Egress goes through the squid sidecar — set HTTPS_PROXY so curl/anthropic-sdk
# routes via 127.0.0.1:3128 (shared network namespace).
ENV HTTPS_PROXY=http://127.0.0.1:3128 \
    HTTP_PROXY=http://127.0.0.1:3128 \
    NO_PROXY=localhost,127.0.0.1

# ANTHROPIC_API_KEY is injected at `docker run -e ...` time — never baked in.
ENV CLAUDE_TEMPERATURE="0.2"

ENTRYPOINT ["python3.13", "/usr/local/bin/laureum-runner"]
