| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2866/fibonacci-6de51de777610f8a19f2390b4dcc1c84db22e7a0.md) | 4,007 |  12,000,265 |  1,151 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2866/keccak-6de51de777610f8a19f2390b4dcc1c84db22e7a0.md) | 21,890 |  18,655,329 |  4,642 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2866/sha2_bench-6de51de777610f8a19f2390b4dcc1c84db22e7a0.md) | 9,497 |  14,793,960 |  1,829 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2866/regex-6de51de777610f8a19f2390b4dcc1c84db22e7a0.md) | 1,512 |  4,137,067 |  428 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2866/ecrecover-6de51de777610f8a19f2390b4dcc1c84db22e7a0.md) | 602 |  123,583 |  276 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2866/pairing-6de51de777610f8a19f2390b4dcc1c84db22e7a0.md) | 941 |  1,745,757 |  307 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2866/kitchen_sink-6de51de777610f8a19f2390b4dcc1c84db22e7a0.md) | 4,166 |  2,579,903 |  892 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/6de51de777610f8a19f2390b4dcc1c84db22e7a0

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27300805895)
