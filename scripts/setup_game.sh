#!/usr/bin/env bash
# Fetch the Cookie Clicker mirror (v2.058) into vendor/. Not committed (proprietary).
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -f vendor/cookieclicker/main.js ]; then
  echo "vendor/cookieclicker already present"
else
  git clone --depth 1 https://github.com/ozh/cookieclicker.git vendor/cookieclicker
fi
grep -o 'var VERSION=[0-9.]*' vendor/cookieclicker/index.html
echo "OK"
