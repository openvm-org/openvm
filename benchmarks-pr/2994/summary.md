| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2994/fibonacci-47fd0bdf51ce68cd1b25e6c1d0d93e8b72c3d039.md) | 3,101 |  12,000,265 |  676 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2994/keccak-47fd0bdf51ce68cd1b25e6c1d0d93e8b72c3d039.md) | 16,544 |  18,655,329 |  3,023 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2994/sha2_bench-47fd0bdf51ce68cd1b25e6c1d0d93e8b72c3d039.md) | 9,681 |  14,793,960 |  1,137 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2994/regex-47fd0bdf51ce68cd1b25e6c1d0d93e8b72c3d039.md) | 1,274 |  4,137,067 |  351 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2994/ecrecover-47fd0bdf51ce68cd1b25e6c1d0d93e8b72c3d039.md) | 583 |  123,583 |  288 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2994/pairing-47fd0bdf51ce68cd1b25e6c1d0d93e8b72c3d039.md) | 919 |  1,745,757 |  309 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2994/kitchen_sink-47fd0bdf51ce68cd1b25e6c1d0d93e8b72c3d039.md) | 4,620 |  2,579,903 |  885 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/47fd0bdf51ce68cd1b25e6c1d0d93e8b72c3d039

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29060984210)
