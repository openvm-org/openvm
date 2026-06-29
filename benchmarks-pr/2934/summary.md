| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2934/fibonacci-23de4a11ba2a9def83d7d67967b45abcf45eb841.md) | 1,055 |  4,000,051 |  402 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2934/keccak-23de4a11ba2a9def83d7d67967b45abcf45eb841.md) | 15,951 |  14,365,133 |  3,071 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2934/sha2_bench-23de4a11ba2a9def83d7d67967b45abcf45eb841.md) | 8,335 |  11,167,961 |  1,027 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2934/regex-23de4a11ba2a9def83d7d67967b45abcf45eb841.md) | 1,154 |  4,090,656 |  352 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2934/ecrecover-23de4a11ba2a9def83d7d67967b45abcf45eb841.md) | 436 |  112,210 |  276 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2934/pairing-23de4a11ba2a9def83d7d67967b45abcf45eb841.md) | 587 |  592,827 |  296 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2934/kitchen_sink-23de4a11ba2a9def83d7d67967b45abcf45eb841.md) | 3,854 |  1,979,971 |  862 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/23de4a11ba2a9def83d7d67967b45abcf45eb841

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28402645648)
