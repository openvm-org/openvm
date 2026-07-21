| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3043/fibonacci-3d3b5b15ab3734066fc90e19243af7ac90ee4731.md) | 417 |  4,000,051 |  239 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3043/keccak-3d3b5b15ab3734066fc90e19243af7ac90ee4731.md) | 8,547 |  14,365,133 |  1,516 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3043/sha2_bench-3d3b5b15ab3734066fc90e19243af7ac90ee4731.md) | 4,205 |  11,167,961 |  526 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3043/regex-3d3b5b15ab3734066fc90e19243af7ac90ee4731.md) | 558 |  4,090,656 |  213 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3043/ecrecover-3d3b5b15ab3734066fc90e19243af7ac90ee4731.md) | 222 |  112,210 |  182 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3043/pairing-3d3b5b15ab3734066fc90e19243af7ac90ee4731.md) | 283 |  592,827 |  181 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3043/kitchen_sink-3d3b5b15ab3734066fc90e19243af7ac90ee4731.md) | 1,930 |  1,979,971 |  468 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/3d3b5b15ab3734066fc90e19243af7ac90ee4731

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29818327261)
