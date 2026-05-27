| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2817/fibonacci-3942296a4c486716eea035d8e42231ba6d3edda9.md) | 3,723 |  12,000,265 |  912 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2817/keccak-3942296a4c486716eea035d8e42231ba6d3edda9.md) | 18,185 |  18,655,329 |  3,317 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2817/sha2_bench-3942296a4c486716eea035d8e42231ba6d3edda9.md) | 9,919 |  14,793,960 |  1,439 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2817/regex-3942296a4c486716eea035d8e42231ba6d3edda9.md) | 1,398 |  4,137,067 |  357 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2817/ecrecover-3942296a4c486716eea035d8e42231ba6d3edda9.md) | 604 |  123,583 |  255 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2817/pairing-3942296a4c486716eea035d8e42231ba6d3edda9.md) | 888 |  1,745,757 |  259 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2817/kitchen_sink-3942296a4c486716eea035d8e42231ba6d3edda9.md) | 1,863 |  2,579,903 |  415 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/3942296a4c486716eea035d8e42231ba6d3edda9

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26532518406)
