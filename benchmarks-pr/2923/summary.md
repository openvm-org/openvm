| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2923/fibonacci-be4794d33623f392feb413b4c86128122f3b4b48.md) | 1,017 |  4,000,051 |  389 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2923/keccak-be4794d33623f392feb413b4c86128122f3b4b48.md) | 16,304 |  14,365,133 |  3,026 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2923/sha2_bench-be4794d33623f392feb413b4c86128122f3b4b48.md) | 8,234 |  11,167,961 |  1,006 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2923/regex-be4794d33623f392feb413b4c86128122f3b4b48.md) | 1,216 |  4,090,656 |  358 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2923/ecrecover-be4794d33623f392feb413b4c86128122f3b4b48.md) | 436 |  112,210 |  281 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2923/pairing-be4794d33623f392feb413b4c86128122f3b4b48.md) | 596 |  592,827 |  295 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2923/kitchen_sink-be4794d33623f392feb413b4c86128122f3b4b48.md) | 3,942 |  1,979,971 |  874 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/be4794d33623f392feb413b4c86128122f3b4b48

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28055315282)
