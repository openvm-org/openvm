| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-bc2593ed2eda08288d3791eecccf6c7acc5ebd54.md) | 3,846 |  12,000,265 |  953 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-bc2593ed2eda08288d3791eecccf6c7acc5ebd54.md) | 18,608 |  18,655,329 |  3,328 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-bc2593ed2eda08288d3791eecccf6c7acc5ebd54.md) | 1,412 |  4,137,067 |  371 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-bc2593ed2eda08288d3791eecccf6c7acc5ebd54.md) | 648 |  123,583 |  274 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-bc2593ed2eda08288d3791eecccf6c7acc5ebd54.md) | 904 |  1,745,757 |  284 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-bc2593ed2eda08288d3791eecccf6c7acc5ebd54.md) | 2,278 |  2,579,903 |  441 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/bc2593ed2eda08288d3791eecccf6c7acc5ebd54

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24141806383)
