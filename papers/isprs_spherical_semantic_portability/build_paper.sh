#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
template_url="https://isprs.org/documents/orangebook/LATEX/ISPRSguidelines_authors_fullpaper_latex_2024_01_17.zip"
template_sha256="e187bf5d55ebb44a14c31e6ae4319969168dc0305e93d04e9d657aa72b7d079d"
build_dir="$(mktemp -d)"
trap 'rm -rf "$build_dir"' EXIT

curl -L --fail --silent --show-error "$template_url" -o "$build_dir/isprs.zip"
actual_sha256="$(shasum -a 256 "$build_dir/isprs.zip" | awk '{print $1}')"
if [[ "$actual_sha256" != "$template_sha256" ]]; then
  echo "ISPRS template checksum mismatch" >&2
  exit 1
fi

unzip -q "$build_dir/isprs.zip" -d "$build_dir/template"
cp "$build_dir/template/isprs.cls" "$build_dir/isprs.cls"
cp "$build_dir/template/isprs.bst" "$build_dir/isprs.bst"
cp "$script_dir/paper.tex" "$build_dir/paper.tex"
cp "$script_dir/references.bib" "$build_dir/references.bib"
mkdir -p "$build_dir/figures" "$repo_root/output/pdf"

for image_name in \
  spherical-cam-imagenet-lakeside.jpg \
  spherical-cam-places365-forest.jpg \
  spherical-cam-openclip-path.jpg \
  spherical-cam-imagenet-studio-desk.jpg \
  spherical-cam-places365-studio-lobby.jpg \
  spherical-cam-openclip-studio-desk.jpg; do
  cp "$repo_root/docs/_static/tutorials/$image_name" "$build_dir/figures/$image_name"
done

(cd "$build_dir" && tectonic -X compile paper.tex --keep-logs --keep-intermediates)
cp "$build_dir/paper.pdf" "$repo_root/output/pdf/isprs_spherical_semantic_portability.pdf"
echo "$repo_root/output/pdf/isprs_spherical_semantic_portability.pdf"
