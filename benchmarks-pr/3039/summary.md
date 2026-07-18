| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3039/fibonacci-b578c91459a5766350bf923c8aee1c17eac44f5f.md) | 415 |  4,000,051 |  236 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3039/keccak-b578c91459a5766350bf923c8aee1c17eac44f5f.md) | 8,571 |  14,365,133 |  1,516 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3039/sha2_bench-b578c91459a5766350bf923c8aee1c17eac44f5f.md) | 4,209 |  11,167,961 |  521 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3039/regex-b578c91459a5766350bf923c8aee1c17eac44f5f.md) | 575 |  4,090,656 |  212 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3039/ecrecover-b578c91459a5766350bf923c8aee1c17eac44f5f.md) | 220 |  112,210 |  182 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3039/pairing-b578c91459a5766350bf923c8aee1c17eac44f5f.md) | 297 |  592,827 |  185 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3039/kitchen_sink-b578c91459a5766350bf923c8aee1c17eac44f5f.md) | 1,920 |  1,979,971 |  459 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/b578c91459a5766350bf923c8aee1c17eac44f5f

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29650331187)
