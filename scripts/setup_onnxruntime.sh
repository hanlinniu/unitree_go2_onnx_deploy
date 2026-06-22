#!/usr/bin/env bash
# Download ONNX Runtime into deploy/thirdparty/
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD="${ROOT}/deploy/thirdparty"
ARCH="$(uname -m)"
VERSION="1.23.2"

mkdir -p "${THIRD}"
cd "${THIRD}"

case "${ARCH}" in
  x86_64)
    TARBALL="onnxruntime-linux-x64-${VERSION}.tgz"
    DIR="onnxruntime-linux-x64-${VERSION}"
    ;;
  aarch64)
    TARBALL="onnxruntime-linux-aarch64-${VERSION}.tgz"
    DIR="onnxruntime-linux-aarch64-${VERSION}"
    ;;
  *)
    echo "Unsupported architecture: ${ARCH}" >&2
    exit 1
    ;;
esac

URL="https://github.com/microsoft/onnxruntime/releases/download/v${VERSION}/${TARBALL}"

if [[ -d "${DIR}" ]]; then
  echo "Already present: ${THIRD}/${DIR}"
  exit 0
fi

echo "Downloading ${URL}"
curl -L --retry 3 -o "${TARBALL}" "${URL}"
tar -xzf "${TARBALL}"
rm -f "${TARBALL}"
echo "Installed ${THIRD}/${DIR}"
