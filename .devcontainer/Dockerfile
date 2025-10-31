# syntax=docker/dockerfile:1

ARG VARIANT="3.12"
ARG USERNAME="vscode"
FROM mcr.microsoft.com/devcontainers/python:${VARIANT}

ENV DEBIAN_FRONTEND="noninteractive"

RUN apt-get update \
  && apt-get install -y --no-install-recommends curl git \
  && rm -rf /var/lib/apt/lists/*

# Install uv globally so both root and the devcontainer user can invoke it.
RUN curl -LsSf https://astral.sh/uv/install.sh \
  | sh -s -- --yes --bin-dir /usr/local/bin

RUN mkdir -p /home/${USERNAME}/.cache/uv \
  && chown -R ${USERNAME}:${USERNAME} /home/${USERNAME}/.cache

RUN cat <<'EOF' >/usr/local/share/load-dotenv.sh \
  && chmod 0755 /usr/local/share/load-dotenv.sh
#!/bin/sh
if [ -z "${DEVCONTAINER_WORKSPACE_DIR}" ]; then
  DEVCONTAINER_WORKSPACE_DIR="/workspaces/llm-rationale-benchmark"
fi
ENV_FILE="${DEVCONTAINER_WORKSPACE_DIR}/.env"
if [ -f "${ENV_FILE}" ]; then
  set -a
  # shellcheck disable=SC1090
  . "${ENV_FILE}"
  set +a
fi
EOF \
  && cat <<'EOF' >/etc/profile.d/00-load-dotenv.sh \
  && chmod 0644 /etc/profile.d/00-load-dotenv.sh
. /usr/local/share/load-dotenv.sh
EOF \
  && mkdir -p /etc/zsh \
  && cat <<'EOF' >/etc/zsh/zshenv
. /usr/local/share/load-dotenv.sh
EOF

WORKDIR /workspaces/llm-rationale-benchmark
