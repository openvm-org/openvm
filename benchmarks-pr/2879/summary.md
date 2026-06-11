| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2879/fibonacci-2f0e31a574212bc8961bb4e43c482e8aba89e617.md) | 3,976 |  12,000,265 |  1,147 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2879/keccak-2f0e31a574212bc8961bb4e43c482e8aba89e617.md) | 21,906 |  18,655,329 |  4,643 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2879/sha2_bench-2f0e31a574212bc8961bb4e43c482e8aba89e617.md) | 9,542 |  14,793,960 |  1,827 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2879/regex-2f0e31a574212bc8961bb4e43c482e8aba89e617.md) | 1,494 |  4,137,067 |  426 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2879/ecrecover-2f0e31a574212bc8961bb4e43c482e8aba89e617.md) | 600 |  123,583 |  283 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2879/pairing-2f0e31a574212bc8961bb4e43c482e8aba89e617.md) | 947 |  1,745,757 |  305 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2879/kitchen_sink-2f0e31a574212bc8961bb4e43c482e8aba89e617.md) | 4,146 |  2,579,903 |  882 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/2f0e31a574212bc8961bb4e43c482e8aba89e617

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27377297132)
