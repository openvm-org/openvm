| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/fibonacci-15fe349026c17ebe0ffeff86ba692bfa047a7351.md) | 498 |  4,000,051 |  230 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/keccak-15fe349026c17ebe0ffeff86ba692bfa047a7351.md) | 7,415 |  14,365,133 |  1,522 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/sha2_bench-15fe349026c17ebe0ffeff86ba692bfa047a7351.md) | 4,166 |  11,167,961 |  530 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/regex-15fe349026c17ebe0ffeff86ba692bfa047a7351.md) | 675 |  4,090,656 |  211 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/ecrecover-15fe349026c17ebe0ffeff86ba692bfa047a7351.md) | 231 |  112,210 |  180 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/pairing-15fe349026c17ebe0ffeff86ba692bfa047a7351.md) | 244 |  592,827 |  183 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3074/kitchen_sink-15fe349026c17ebe0ffeff86ba692bfa047a7351.md) | 2,045 |  1,979,971 |  457 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/15fe349026c17ebe0ffeff86ba692bfa047a7351

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30344496789)
