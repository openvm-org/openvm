| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2806/fibonacci-ae2ebbb2729b9be7001cd6ef6bb4c591bcf42abd.md) | 3,750 |  12,000,265 |  914 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2806/keccak-ae2ebbb2729b9be7001cd6ef6bb4c591bcf42abd.md) | 18,551 |  18,655,329 |  3,285 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2806/sha2_bench-ae2ebbb2729b9be7001cd6ef6bb4c591bcf42abd.md) | 10,128 |  14,793,960 |  1,452 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2806/regex-ae2ebbb2729b9be7001cd6ef6bb4c591bcf42abd.md) | 1,403 |  4,137,067 |  354 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2806/ecrecover-ae2ebbb2729b9be7001cd6ef6bb4c591bcf42abd.md) | 598 |  123,583 |  252 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2806/pairing-ae2ebbb2729b9be7001cd6ef6bb4c591bcf42abd.md) | 890 |  1,745,757 |  263 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2806/kitchen_sink-ae2ebbb2729b9be7001cd6ef6bb4c591bcf42abd.md) | 1,905 |  2,579,903 |  410 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/ae2ebbb2729b9be7001cd6ef6bb4c591bcf42abd

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26279972401)
