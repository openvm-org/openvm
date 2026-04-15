| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2710/fibonacci-d3ad8525a321e4c1715f1df7e5b3c1e032af9fea.md) | 3,819 |  12,000,265 |  945 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2710/keccak-d3ad8525a321e4c1715f1df7e5b3c1e032af9fea.md) | 18,819 |  18,655,329 |  3,364 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2710/sha2_bench-d3ad8525a321e4c1715f1df7e5b3c1e032af9fea.md) | 9,043 |  14,793,960 |  1,397 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2710/regex-d3ad8525a321e4c1715f1df7e5b3c1e032af9fea.md) | 1,424 |  4,137,067 |  376 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2710/ecrecover-d3ad8525a321e4c1715f1df7e5b3c1e032af9fea.md) | 647 |  123,583 |  271 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2710/pairing-d3ad8525a321e4c1715f1df7e5b3c1e032af9fea.md) | 915 |  1,745,757 |  290 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2710/kitchen_sink-d3ad8525a321e4c1715f1df7e5b3c1e032af9fea.md) | 2,092 |  2,579,903 |  434 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/d3ad8525a321e4c1715f1df7e5b3c1e032af9fea

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24477077549)
