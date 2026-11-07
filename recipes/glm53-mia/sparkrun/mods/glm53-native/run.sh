#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 verify.py
install -d /opt/spark-inference/glm53-mia
install -m 0644 chat_template.jinja /opt/spark-inference/glm53-mia/chat_template.jinja
install -m 0755 capacity.py readiness.py /opt/spark-inference/glm53-mia/
install -m 0755 serve.py /usr/local/bin/tensorfold-entrypoint
