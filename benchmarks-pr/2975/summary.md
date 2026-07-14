| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2975/fibonacci-217f047542b4cc5a60b5019e76f9016adf7aadd1.md) | 469 |  4,000,051 |  230 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2975/keccak-217f047542b4cc5a60b5019e76f9016adf7aadd1.md) | 8,752 |  14,365,133 |  1,519 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2975/sha2_bench-217f047542b4cc5a60b5019e76f9016adf7aadd1.md) | 3,910 |  11,167,961 |  516 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2975/regex-217f047542b4cc5a60b5019e76f9016adf7aadd1.md) | 500 |  4,090,656 |  189 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2975/ecrecover-217f047542b4cc5a60b5019e76f9016adf7aadd1.md) | 223 |  112,210 |  181 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2975/pairing-217f047542b4cc5a60b5019e76f9016adf7aadd1.md) | 261 |  592,827 |  184 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2975/kitchen_sink-217f047542b4cc5a60b5019e76f9016adf7aadd1.md) | 1,927 |  1,979,971 |  469 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/217f047542b4cc5a60b5019e76f9016adf7aadd1

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29375031734)
