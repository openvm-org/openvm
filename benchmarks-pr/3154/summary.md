| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3154/fibonacci-7a52b9a219e0fde4f06e251b94e9a3b1c3ecb6cc.md) | 469 |  4,000,051 |  233 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3154/keccak-7a52b9a219e0fde4f06e251b94e9a3b1c3ecb6cc.md) | 7,678 |  14,365,133 |  1,656 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3154/sha2_bench-7a52b9a219e0fde4f06e251b94e9a3b1c3ecb6cc.md) | 4,250 |  11,167,961 |  532 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3154/regex-7a52b9a219e0fde4f06e251b94e9a3b1c3ecb6cc.md) | 746 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3154/ecrecover-7a52b9a219e0fde4f06e251b94e9a3b1c3ecb6cc.md) | 210 |  112,210 |  194 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3154/pairing-7a52b9a219e0fde4f06e251b94e9a3b1c3ecb6cc.md) | 255 |  592,827 |  173 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3154/kitchen_sink-7a52b9a219e0fde4f06e251b94e9a3b1c3ecb6cc.md) | 2,232 |  1,979,971 |  473 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/7a52b9a219e0fde4f06e251b94e9a3b1c3ecb6cc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/34606076840)
