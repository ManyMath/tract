#!/usr/bin/env bash
# Builds the size-optimized tract_ffi.wasm: cargo's `wasm` profile, then
# wasm-opt -Oz, then checks that the module has no imports and fits the budget.
#
# Usage: tool/build_wasm.sh [OUT]   (default: packages/tract_wasm/tract_ffi.wasm)
#
# Environment:
#   WASM_OPT             wasm-opt binary (default: wasm-opt on PATH, else ~/.cargo/bin)
#   TRACT_WASM_MAX_BYTES        size budget for the optimized module (default 8500000)
#   TRACT_WASM_MAX_GZIP_BYTES   size budget for its gzip -9 form (default 2600000)
set -euo pipefail

native_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ $# -ge 1 ]; then
  # Resolve against the caller's cwd, before the cd below.
  case "$1" in
    /*) out="$1" ;;
    *) out="$PWD/$1" ;;
  esac
else
  out="$native_dir/../../tract_wasm/tract_ffi.wasm"
fi
max_bytes="${TRACT_WASM_MAX_BYTES:-8500000}"
max_gzip="${TRACT_WASM_MAX_GZIP_BYTES:-2600000}"
wasm_opt="${WASM_OPT:-$(command -v wasm-opt || echo "$HOME/.cargo/bin/wasm-opt")}"

cd "$native_dir"
cargo build --profile wasm --target wasm32-unknown-unknown
built="${CARGO_TARGET_DIR:-target}/wasm32-unknown-unknown/wasm/tract_ffi.wasm"

# rustc enables these by default for wasm32-unknown-unknown, and tract's
# kernels use simd128; wasm-opt must be told so it accepts the module.
mkdir -p "$(dirname "$out")"
"$wasm_opt" -Oz \
  --enable-bulk-memory --enable-sign-ext --enable-nontrapping-float-to-int \
  --enable-mutable-globals --enable-multivalue --enable-reference-types \
  --enable-simd \
  "$built" -o "$out"

node -e '
const m = new WebAssembly.Module(require("fs").readFileSync(process.argv[1]));
const imports = WebAssembly.Module.imports(m);
if (imports.length) {
  console.error("tract_ffi.wasm must have zero imports, found:", JSON.stringify(imports));
  process.exit(1);
}
' "$out"

size=$(wc -c < "$out")
gz=$(gzip -9 -c "$out" | wc -c)
echo "tract_ffi.wasm: $size bytes, $gz bytes gzipped (budgets: $max_bytes / $max_gzip), 0 imports"
if [ "$size" -gt "$max_bytes" ] || [ "$gz" -gt "$max_gzip" ]; then
  echo "size budget exceeded" >&2
  exit 1
fi
