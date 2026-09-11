| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3155/fibonacci-bfff2835a7abc0e89a81133c9cfb6a21776e4d98.md) | 489 |  4,000,051 |  235 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3155/keccak-bfff2835a7abc0e89a81133c9cfb6a21776e4d98.md) | 7,668 |  14,365,133 |  1,598 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3155/sha2_bench-bfff2835a7abc0e89a81133c9cfb6a21776e4d98.md) | 4,361 |  11,167,961 |  521 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3155/regex-bfff2835a7abc0e89a81133c9cfb6a21776e4d98.md) | 741 |  4,090,656 |  218 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3155/ecrecover-bfff2835a7abc0e89a81133c9cfb6a21776e4d98.md) | 207 |  112,210 |  188 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3155/pairing-bfff2835a7abc0e89a81133c9cfb6a21776e4d98.md) | 251 |  592,827 |  173 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3155/kitchen_sink-bfff2835a7abc0e89a81133c9cfb6a21776e4d98.md) | 2,253 |  1,979,971 |  473 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/bfff2835a7abc0e89a81133c9cfb6a21776e4d98

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/34580352626)
