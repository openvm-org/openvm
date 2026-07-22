| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/fibonacci-c93e5dcf73c9a2f86a19dfa0ab7260fb8fbceb4f.md) | 483 |  4,000,051 |  244 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/keccak-c93e5dcf73c9a2f86a19dfa0ab7260fb8fbceb4f.md) | 7,315 |  14,365,133 |  1,528 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/sha2_bench-c93e5dcf73c9a2f86a19dfa0ab7260fb8fbceb4f.md) | 4,758 |  11,167,961 |  528 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/regex-c93e5dcf73c9a2f86a19dfa0ab7260fb8fbceb4f.md) | 678 |  4,090,656 |  218 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/ecrecover-c93e5dcf73c9a2f86a19dfa0ab7260fb8fbceb4f.md) | 232 |  112,210 |  185 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/pairing-c93e5dcf73c9a2f86a19dfa0ab7260fb8fbceb4f.md) | 329 |  592,827 |  190 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3061/kitchen_sink-c93e5dcf73c9a2f86a19dfa0ab7260fb8fbceb4f.md) | 2,672 |  1,979,971 |  470 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/c93e5dcf73c9a2f86a19dfa0ab7260fb8fbceb4f

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29960719889)
