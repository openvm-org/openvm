| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3043/fibonacci-c1d72d1985356a21cf802ea9dd42ea56cd4dd0d6.md) | 410 |  4,000,051 |  236 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3043/keccak-c1d72d1985356a21cf802ea9dd42ea56cd4dd0d6.md) | 8,564 |  14,365,133 |  1,519 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3043/sha2_bench-c1d72d1985356a21cf802ea9dd42ea56cd4dd0d6.md) | 4,229 |  11,167,961 |  522 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3043/regex-c1d72d1985356a21cf802ea9dd42ea56cd4dd0d6.md) | 569 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3043/ecrecover-c1d72d1985356a21cf802ea9dd42ea56cd4dd0d6.md) | 227 |  112,210 |  185 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3043/pairing-c1d72d1985356a21cf802ea9dd42ea56cd4dd0d6.md) | 295 |  592,827 |  183 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3043/kitchen_sink-c1d72d1985356a21cf802ea9dd42ea56cd4dd0d6.md) | 1,940 |  1,979,971 |  470 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/c1d72d1985356a21cf802ea9dd42ea56cd4dd0d6

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29816708942)
