| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/fibonacci-721fb15c5348fef6ec565fb62784443f41179104.md) | 461 |  4,000,051 |  228 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/keccak-721fb15c5348fef6ec565fb62784443f41179104.md) | 7,417 |  14,365,133 |  1,535 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/sha2_bench-721fb15c5348fef6ec565fb62784443f41179104.md) | 4,158 |  11,167,961 |  519 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/regex-721fb15c5348fef6ec565fb62784443f41179104.md) | 665 |  4,090,656 |  214 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/ecrecover-721fb15c5348fef6ec565fb62784443f41179104.md) | 196 |  112,210 |  197 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/pairing-721fb15c5348fef6ec565fb62784443f41179104.md) | 234 |  592,827 |  196 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/kitchen_sink-721fb15c5348fef6ec565fb62784443f41179104.md) | 2,030 |  1,979,971 |  523 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/721fb15c5348fef6ec565fb62784443f41179104

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31615168338)
