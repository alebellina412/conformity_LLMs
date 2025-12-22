#!/usr/bin/env bash
set -euo pipefail

DOI="10.5281/zenodo.18022032"
RECORD_URL="https://doi.org/${DOI}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK_DIR="${ROOT_DIR}/.zenodo_download"

mkdir -p "${WORK_DIR}"

echo "Downloading Zenodo dataset from ${RECORD_URL}"
echo "Files will be extracted into ${ROOT_DIR}"

if command -v curl >/dev/null 2>&1; then
  curl -L "${RECORD_URL}" -o "${WORK_DIR}/record.html"
elif command -v wget >/dev/null 2>&1; then
  wget -O "${WORK_DIR}/record.html" "${RECORD_URL}"
else
  echo "Error: curl or wget is required to download the dataset." >&2
  exit 1
fi

FILE_URLS=()
if command -v python3 >/dev/null 2>&1; then
  mapfile -t FILE_URLS < <(python3 - <<'PY'
import re
from pathlib import Path

html = Path(".zenodo_download/record.html").read_text(encoding="utf-8", errors="ignore")
urls = re.findall(r'href="(https://zenodo\\.org/records/[^"]+/files/[^"]+)"', html)
for url in sorted(set(urls)):
    print(url)
PY
  )
else
  while IFS= read -r line; do
    case "$line" in
      *zenodo.org/records/*/files/*)
        FILE_URLS+=("${line}")
        ;;
    esac
  done < <(sed -n 's/.*href="\\(https:\\/\\/zenodo\\.org\\/records\\/[^"]*\\/files\\/[^"]*\\)".*/\\1/p' "${WORK_DIR}/record.html" | sort -u)
fi

if [[ "${#FILE_URLS[@]}" -eq 0 ]]; then
  echo "Error: could not find downloadable files in the Zenodo record." >&2
  exit 1
fi

for url in "${FILE_URLS[@]}"; do
  filename="${url##*/}"
  dest="${WORK_DIR}/${filename}"
  echo "Downloading ${filename}"
  if command -v curl >/dev/null 2>&1; then
    curl -L "${url}" -o "${dest}"
  else
    wget -O "${dest}" "${url}"
  fi
done

for archive in "${WORK_DIR}"/*; do
  case "${archive}" in
    *.zip)
      echo "Extracting ${archive}"
      unzip -o "${archive}" -d "${ROOT_DIR}" >/dev/null
      ;;
    *.tar.gz|*.tgz)
      echo "Extracting ${archive}"
      tar -xzf "${archive}" -C "${ROOT_DIR}"
      ;;
    *.tar)
      echo "Extracting ${archive}"
      tar -xf "${archive}" -C "${ROOT_DIR}"
      ;;
    *)
      echo "Skipping ${archive} (not an archive)"
      ;;
  esac
done

echo "Done. images/ and data/ should now be available in ${ROOT_DIR}."
