| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/fibonacci-6fb2ee731aa06a80aa92c388413f7eb19d8c95e3.md) | 473 |  4,000,051 |  231 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/keccak-6fb2ee731aa06a80aa92c388413f7eb19d8c95e3.md) | 7,347 |  14,365,133 |  1,511 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/sha2_bench-6fb2ee731aa06a80aa92c388413f7eb19d8c95e3.md) | 4,124 |  11,167,961 |  520 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/regex-6fb2ee731aa06a80aa92c388413f7eb19d8c95e3.md) | 658 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/ecrecover-6fb2ee731aa06a80aa92c388413f7eb19d8c95e3.md) | 225 |  112,210 |  181 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/pairing-6fb2ee731aa06a80aa92c388413f7eb19d8c95e3.md) | 231 |  592,827 |  185 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/kitchen_sink-6fb2ee731aa06a80aa92c388413f7eb19d8c95e3.md) | 2,035 |  1,979,971 |  462 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/6fb2ee731aa06a80aa92c388413f7eb19d8c95e3

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31041891974)
