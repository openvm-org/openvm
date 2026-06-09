| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2861/fibonacci-05593461d1ac561fb4acaa8e775c864439186237.md) | 3,956 |  12,000,265 |  1,146 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2861/keccak-05593461d1ac561fb4acaa8e775c864439186237.md) | 21,596 |  18,655,329 |  4,575 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2861/sha2_bench-05593461d1ac561fb4acaa8e775c864439186237.md) | 9,700 |  14,793,960 |  1,853 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2861/regex-05593461d1ac561fb4acaa8e775c864439186237.md) | 1,507 |  4,137,067 |  429 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2861/ecrecover-05593461d1ac561fb4acaa8e775c864439186237.md) | 611 |  123,583 |  283 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2861/pairing-05593461d1ac561fb4acaa8e775c864439186237.md) | 957 |  1,745,757 |  316 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2861/kitchen_sink-05593461d1ac561fb4acaa8e775c864439186237.md) | 4,109 |  2,579,903 |  875 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/05593461d1ac561fb4acaa8e775c864439186237

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27189634741)
