| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/fibonacci-36a796fde38bed43466ab480e9c6d294130d5d99.md) | 1,021 |  4,000,051 |  385 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/keccak-36a796fde38bed43466ab480e9c6d294130d5d99.md) | 15,623 |  14,365,133 |  3,054 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/sha2_bench-36a796fde38bed43466ab480e9c6d294130d5d99.md) | 7,858 |  11,167,961 |  1,008 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/regex-36a796fde38bed43466ab480e9c6d294130d5d99.md) | 1,015 |  4,090,656 |  286 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/ecrecover-36a796fde38bed43466ab480e9c6d294130d5d99.md) | 443 |  112,210 |  281 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/pairing-36a796fde38bed43466ab480e9c6d294130d5d99.md) | 558 |  592,827 |  295 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2922/kitchen_sink-36a796fde38bed43466ab480e9c6d294130d5d99.md) | 3,840 |  1,979,971 |  873 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/36a796fde38bed43466ab480e9c6d294130d5d99

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28258731602)
