import * as wasm from "wasm-test";

try {
  wasm.prove_fib();
} catch (e) {
  console.error(e);
}
