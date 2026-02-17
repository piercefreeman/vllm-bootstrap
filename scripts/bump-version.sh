#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <tag>" >&2
    echo "Example: $0 v0.2.0" >&2
    exit 1
fi

TAG="$1"
VERSION="${TAG#v}"

# Validate version looks like semver
if ! [[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+.*$ ]]; then
    echo "Error: '$TAG' does not look like a valid version tag (expected vX.Y.Z)" >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

for pyproject in "$REPO_ROOT/pyproject.toml" "$REPO_ROOT/vllm-bootstrap-client/pyproject.toml"; do
    if [ ! -f "$pyproject" ]; then
        echo "Warning: $pyproject not found, skipping" >&2
        continue
    fi
    sed -i.bak "s/^version = \".*\"/version = \"$VERSION\"/" "$pyproject"
    rm -f "${pyproject}.bak"
    echo "Updated $pyproject to version $VERSION"
done
