| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/fibonacci-39645effcb20584e14c22cc9993b040db9cf6127.md) | 469 |  4,000,051 |  244 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/keccak-39645effcb20584e14c22cc9993b040db9cf6127.md) | 7,329 |  14,365,133 |  1,574 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/sha2_bench-39645effcb20584e14c22cc9993b040db9cf6127.md) | 4,787 |  11,167,961 |  534 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/regex-39645effcb20584e14c22cc9993b040db9cf6127.md) | 673 |  4,090,656 |  221 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/ecrecover-39645effcb20584e14c22cc9993b040db9cf6127.md) | 232 |  112,210 |  184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/pairing-39645effcb20584e14c22cc9993b040db9cf6127.md) | 312 |  592,827 |  188 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3058/kitchen_sink-39645effcb20584e14c22cc9993b040db9cf6127.md) | 2,664 |  1,979,971 |  473 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/39645effcb20584e14c22cc9993b040db9cf6127

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30132944114)
