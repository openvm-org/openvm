| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2909/fibonacci-60ba0c82246d71d86ada84db7b35021977da0697.md) | 3,013 |  12,000,265 |  664 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2909/keccak-60ba0c82246d71d86ada84db7b35021977da0697.md) | 16,521 |  18,655,329 |  3,066 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2909/sha2_bench-60ba0c82246d71d86ada84db7b35021977da0697.md) | 9,144 |  14,793,960 |  1,123 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2909/regex-60ba0c82246d71d86ada84db7b35021977da0697.md) | 1,168 |  4,137,067 |  354 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2909/ecrecover-60ba0c82246d71d86ada84db7b35021977da0697.md) | 599 |  123,583 |  283 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2909/pairing-60ba0c82246d71d86ada84db7b35021977da0697.md) | 931 |  1,745,757 |  305 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2909/kitchen_sink-60ba0c82246d71d86ada84db7b35021977da0697.md) | 4,106 |  2,579,903 |  880 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/60ba0c82246d71d86ada84db7b35021977da0697

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27827447712)
