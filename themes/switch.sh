#!/bin/sh
# Switch the site between the summer and winter look.
# Usage (from anywhere):  sh themes/switch.sh summer   or   sh themes/switch.sh winter
# Then check `git diff`, commit and push. macOS (BSD sed) syntax.
set -e
t="$1"
case "$t" in summer|winter) ;; *) echo "usage: sh themes/switch.sh summer|winter"; exit 1 ;; esac
cd "$(dirname "$0")/.."
sed -i '' -E "s#@import url\('themes/(summer|winter)\.css'\);#@import url('themes/$t.css');#" style.css
cp "themes/favicon-$t.svg" favicon.svg
v="$t$(date +%Y%m%d)"
find . -name '*.html' -not -path './slides/*' -not -path './notebooks/*' \
  -exec sed -i '' -E "s#favicon\.svg(\?v=[0-9a-z]+)?\"#favicon.svg?v=$v\"#" {} +
echo "Switched to $t theme (favicon cache key v=$v). Review with git diff, then commit and push."
