#!/usr/bin/env bash
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ramulator="$repo/src/dram/ramulator2"
if [ ! -f "$ramulator/CMakeLists.txt" ]; then
  git -C "$repo" submodule update --init --recursive
fi

# The upstream PIM patch adds this file. Its reverse-apply check conflicts
# with the later GCC header patch because both touch src/base/utils.h.
if [ -f "$ramulator/src/dram_controller/impl/PIM_dram_controller.cpp" ]; then
  echo "Upstream Ramulator PIM patch already applied"
else
  git -C "$ramulator" apply --check "$repo/patch/ramulator2_pim.patch"
  git -C "$ramulator" apply "$repo/patch/ramulator2_pim.patch"
fi

compat="$repo/midsem/ramulator-gcc15-cstdint.patch"
if git -C "$ramulator" apply --reverse --check --ignore-space-change "$compat" >/dev/null 2>&1; then
  echo "GCC header compatibility patch already applied"
else
  git -C "$ramulator" apply --check --ignore-space-change "$compat"
  git -C "$ramulator" apply --ignore-space-change "$compat"
fi

cmake -S "$repo" -B "$repo/build" -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build "$repo/build" --parallel 4

for entry in \
  "offchip:config_offchip.yaml" \
  "offchip_bw16:config_offchip_bw16.yaml" \
  "offchip_bw64:config_offchip_bw64.yaml"; do
  folder="${entry%%:*}"
  config="${entry#*:}"
  out="$repo/midsem/artifacts/$folder"
  mkdir -p "$out" "$repo/data/$folder"
  cp "$repo/$config" "$out/config.yaml"
  (
    cd "$repo/build"
    MOE_TRACE_PATH="$out/routes.csv" timeout --signal=TERM 900 \
      ./run "../$config" >"$out/stdout.log" 2>"$out/stderr.log"
  )
  printf '0\n' >"$out/exit_code.txt"
  cp "$repo/data/$folder/"*decode.csv "$out/"
  echo "Completed $folder"
done

echo "Run python3 midsem/analyze_midsem.py to regenerate the timing plots."
echo "Run python3 midsem/analyze_real_trace.py for the separate public trace."
