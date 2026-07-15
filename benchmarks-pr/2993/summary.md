| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/fibonacci-79c1fe8b799abbd1e6bd2e169b37099135f89917.md) | 466 |  4,000,051 |  230 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/keccak-79c1fe8b799abbd1e6bd2e169b37099135f89917.md) | 8,665 |  14,365,133 |  1,516 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/sha2_bench-79c1fe8b799abbd1e6bd2e169b37099135f89917.md) | 4,031 |  11,167,961 |  521 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/regex-79c1fe8b799abbd1e6bd2e169b37099135f89917.md) | 566 |  4,090,656 |  217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/ecrecover-79c1fe8b799abbd1e6bd2e169b37099135f89917.md) | 222 |  112,210 |  184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/pairing-79c1fe8b799abbd1e6bd2e169b37099135f89917.md) | 278 |  592,827 |  185 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2993/kitchen_sink-79c1fe8b799abbd1e6bd2e169b37099135f89917.md) | 1,933 |  1,979,971 |  457 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/79c1fe8b799abbd1e6bd2e169b37099135f89917

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29398571732)
