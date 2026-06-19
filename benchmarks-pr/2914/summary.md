| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2914/fibonacci-958898bf1f195821fec161cb6be37aa88c0470fa.md) | 3,058 |  12,000,265 |  672 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2914/keccak-958898bf1f195821fec161cb6be37aa88c0470fa.md) | 16,408 |  18,655,329 |  3,053 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2914/sha2_bench-958898bf1f195821fec161cb6be37aa88c0470fa.md) | 9,091 |  14,793,960 |  1,113 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2914/regex-958898bf1f195821fec161cb6be37aa88c0470fa.md) | 1,174 |  4,137,067 |  354 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2914/ecrecover-958898bf1f195821fec161cb6be37aa88c0470fa.md) | 602 |  123,583 |  285 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2914/pairing-958898bf1f195821fec161cb6be37aa88c0470fa.md) | 933 |  1,745,757 |  305 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2914/kitchen_sink-958898bf1f195821fec161cb6be37aa88c0470fa.md) | 4,116 |  2,579,903 |  888 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/958898bf1f195821fec161cb6be37aa88c0470fa

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27847687283)
