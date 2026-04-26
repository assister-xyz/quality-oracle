# QO-059 egress-proxy sidecar for the L3 Claude Code Docker harness.
#
# Pattern (replaces the original `--network=none` design from R7):
#   1. Main container runs `--network=container:laureum-proxy`.
#   2. This sidecar holds the only bridge interface for that pair.
#   3. squid enforces an HTTPS-only allowlist for Anthropic + judge providers.
#   4. Any skill-issued `Bash('curl evil.com')` traverses squid and is denied;
#      the denial surfaces in `tool_calls_log` for SKILL-POISON-02 detection.
#
# AC2 + AC6 — egress restricted to api.anthropic.com, api.cerebras.ai,
# api.groq.com only; everything else 403.
#
# This image is intentionally minimal (Alpine + squid). No application code,
# no shell scripts that exec into the bridge namespace — just squid.

FROM alpine:3.20

RUN apk add --no-cache squid \
    && mkdir -p /var/log/squid /var/cache/squid \
    && chown -R squid:squid /var/log/squid /var/cache/squid

COPY squid.conf /etc/squid/squid.conf

# Squid listens on 3128 (HTTP CONNECT); the main container uses
# HTTPS_PROXY=http://localhost:3128 since they share network namespace.
EXPOSE 3128

# Run in foreground so Docker captures squid's process lifecycle.
USER squid
ENTRYPOINT ["squid", "-N", "-d", "1", "-f", "/etc/squid/squid.conf"]
