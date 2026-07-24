| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3054/fibonacci-d80981b71f518f2998aaae84aff4686edbd3a96a.md) | 473 |  4,000,051 |  242 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3054/keccak-d80981b71f518f2998aaae84aff4686edbd3a96a.md) | 7,337 |  14,365,133 |  1,541 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3054/sha2_bench-d80981b71f518f2998aaae84aff4686edbd3a96a.md) | 4,695 |  11,167,961 |  531 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3054/regex-d80981b71f518f2998aaae84aff4686edbd3a96a.md) | 677 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3054/ecrecover-d80981b71f518f2998aaae84aff4686edbd3a96a.md) | 224 |  112,210 |  185 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3054/pairing-d80981b71f518f2998aaae84aff4686edbd3a96a.md) | 311 |  592,827 |  186 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3054/kitchen_sink-d80981b71f518f2998aaae84aff4686edbd3a96a.md) | 2,668 |  1,979,971 |  466 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/d80981b71f518f2998aaae84aff4686edbd3a96a

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30088117209)
