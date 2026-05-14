| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/fibonacci-723e24c1565612abc6c5dd7ad7dcb38a58a05f78.md) | 1,888 |  4,000,051 |  517 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/keccak-723e24c1565612abc6c5dd7ad7dcb38a58a05f78.md) | 13,518 |  14,365,133 |  2,199 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/sha2_bench-723e24c1565612abc6c5dd7ad7dcb38a58a05f78.md) | 9,460 |  11,167,961 |  1,405 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/regex-723e24c1565612abc6c5dd7ad7dcb38a58a05f78.md) | 1,554 |  4,090,656 |  355 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/ecrecover-723e24c1565612abc6c5dd7ad7dcb38a58a05f78.md) | 598 |  112,210 |  266 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/pairing-723e24c1565612abc6c5dd7ad7dcb38a58a05f78.md) | 739 |  592,827 |  260 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2719/kitchen_sink-723e24c1565612abc6c5dd7ad7dcb38a58a05f78.md) | 1,871 |  1,979,971 |  406 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/723e24c1565612abc6c5dd7ad7dcb38a58a05f78

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25887600087)
