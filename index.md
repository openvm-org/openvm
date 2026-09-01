| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/fibonacci-b99f8d79949ebc1dc471876ee11ed34abba2b673.md) | 1,681 |  12,000,265 |  373 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/keccak-b99f8d79949ebc1dc471876ee11ed34abba2b673.md) | 9,686 |  18,655,329 |  1,560 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/sha2_bench-b99f8d79949ebc1dc471876ee11ed34abba2b673.md) | 5,372 |  14,793,960 |  597 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/regex-b99f8d79949ebc1dc471876ee11ed34abba2b673.md) | 703 |  4,137,067 |  218 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/ecrecover-b99f8d79949ebc1dc471876ee11ed34abba2b673.md) | 435 |  123,583 |  190 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/pairing-b99f8d79949ebc1dc471876ee11ed34abba2b673.md) | 588 |  1,745,757 |  196 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/kitchen_sink-b99f8d79949ebc1dc471876ee11ed34abba2b673.md) | 2,292 |  2,579,903 |  494 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/b99f8d79949ebc1dc471876ee11ed34abba2b673

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/33546604162)
