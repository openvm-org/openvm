#!/bin/bash
set -euo pipefail

dir="$1"
status=0

install_file=$(find "$dir" -name "install.mdx")
tag=$(awk '{
  for (i = 1; i <= NF; i++) {
    if ($i == "--tag" && (i+1) <= NF) {
      print $(i+1)
      exit
    }
  }
}' $install_file)

echo "Tag is $tag"

# Find all regular files under the directory
while IFS= read -r file; do
  awk -v tag="$tag" '
    /^openvm/ && index($0, "github.com/openvm-org/openvm.git") && /\}$/ {
      expected = "tag = \"" tag "\""
      if (index($0, expected) == 0) {
        print FILENAME ": missing tag in line -> " $0
        exit_status = 1
      }
    }
    END { if (exit_status == 1) exit 1 }
  ' "$file" || status=1
done < <(find "$dir" -type f | egrep "md[x]$")

exit $status

