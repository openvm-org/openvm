| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/fibonacci-b3c4a65f631a6fbcdd527661aab4d57a4ccf95b3.md) | 464 |  4,000,051 |  242 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/keccak-b3c4a65f631a6fbcdd527661aab4d57a4ccf95b3.md) | 7,329 |  14,365,133 |  1,541 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/sha2_bench-b3c4a65f631a6fbcdd527661aab4d57a4ccf95b3.md) | 4,759 |  11,167,961 |  532 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/regex-b3c4a65f631a6fbcdd527661aab4d57a4ccf95b3.md) | 659 |  4,090,656 |  218 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/ecrecover-b3c4a65f631a6fbcdd527661aab4d57a4ccf95b3.md) | 223 |  112,210 |  181 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/pairing-b3c4a65f631a6fbcdd527661aab4d57a4ccf95b3.md) | 324 |  592,827 |  187 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3006/kitchen_sink-b3c4a65f631a6fbcdd527661aab4d57a4ccf95b3.md) | 2,656 |  1,979,971 |  473 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/b3c4a65f631a6fbcdd527661aab4d57a4ccf95b3

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29833799192)
