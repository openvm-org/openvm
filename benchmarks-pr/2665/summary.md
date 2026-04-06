| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2665/fibonacci-764c7c092d3d845c399bb9990ef717fa17d49701.md) | 3,820 |  12,000,265 |  955 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2665/keccak-764c7c092d3d845c399bb9990ef717fa17d49701.md) | 18,722 |  18,655,329 |  3,353 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2665/regex-764c7c092d3d845c399bb9990ef717fa17d49701.md) | 1,420 |  4,137,067 |  374 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2665/ecrecover-764c7c092d3d845c399bb9990ef717fa17d49701.md) | 651 |  123,583 |  273 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2665/pairing-764c7c092d3d845c399bb9990ef717fa17d49701.md) | 909 |  1,745,757 |  286 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2665/kitchen_sink-764c7c092d3d845c399bb9990ef717fa17d49701.md) | 2,292 |  2,579,903 |  443 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/764c7c092d3d845c399bb9990ef717fa17d49701

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24056122683)
