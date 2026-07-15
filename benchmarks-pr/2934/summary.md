| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2934/fibonacci-ed6f1f76808925a3aa7f140d77b2dd313b90f49d.md) | 414 |  4,000,051 |  231 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2934/keccak-ed6f1f76808925a3aa7f140d77b2dd313b90f49d.md) | 8,351 |  14,365,133 |  1,513 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2934/sha2_bench-ed6f1f76808925a3aa7f140d77b2dd313b90f49d.md) | 3,976 |  11,167,961 |  530 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2934/regex-ed6f1f76808925a3aa7f140d77b2dd313b90f49d.md) | 566 |  4,090,656 |  212 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2934/ecrecover-ed6f1f76808925a3aa7f140d77b2dd313b90f49d.md) | 221 |  112,210 |  184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2934/pairing-ed6f1f76808925a3aa7f140d77b2dd313b90f49d.md) | 264 |  592,827 |  181 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2934/kitchen_sink-ed6f1f76808925a3aa7f140d77b2dd313b90f49d.md) | 1,907 |  1,979,971 |  467 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/ed6f1f76808925a3aa7f140d77b2dd313b90f49d

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29452018258)
