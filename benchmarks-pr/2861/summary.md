| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2861/fibonacci-ed1f13e4669459eb31bbe3e276db494e07bb2db5.md) | 3,931 |  12,000,265 |  1,138 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2861/keccak-ed1f13e4669459eb31bbe3e276db494e07bb2db5.md) | 21,638 |  18,655,329 |  4,592 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2861/sha2_bench-ed1f13e4669459eb31bbe3e276db494e07bb2db5.md) | 9,468 |  14,793,960 |  1,826 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2861/regex-ed1f13e4669459eb31bbe3e276db494e07bb2db5.md) | 1,520 |  4,137,067 |  430 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2861/ecrecover-ed1f13e4669459eb31bbe3e276db494e07bb2db5.md) | 609 |  123,583 |  287 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2861/pairing-ed1f13e4669459eb31bbe3e276db494e07bb2db5.md) | 948 |  1,745,757 |  309 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2861/kitchen_sink-ed1f13e4669459eb31bbe3e276db494e07bb2db5.md) | 4,130 |  2,579,903 |  881 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/ed1f13e4669459eb31bbe3e276db494e07bb2db5

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27189333859)
