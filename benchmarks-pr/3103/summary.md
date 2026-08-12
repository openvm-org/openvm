| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/fibonacci-3dd50912ebb8fb766d2a249f15e3ce91977744b6.md) | 462 |  4,000,051 |  230 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/keccak-3dd50912ebb8fb766d2a249f15e3ce91977744b6.md) | 7,340 |  14,365,133 |  1,511 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/sha2_bench-3dd50912ebb8fb766d2a249f15e3ce91977744b6.md) | 4,181 |  11,167,961 |  523 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/regex-3dd50912ebb8fb766d2a249f15e3ce91977744b6.md) | 670 |  4,090,656 |  219 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/ecrecover-3dd50912ebb8fb766d2a249f15e3ce91977744b6.md) | 197 |  112,210 |  197 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/pairing-3dd50912ebb8fb766d2a249f15e3ce91977744b6.md) | 233 |  592,827 |  195 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/kitchen_sink-3dd50912ebb8fb766d2a249f15e3ce91977744b6.md) | 2,048 |  1,979,971 |  531 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/3dd50912ebb8fb766d2a249f15e3ce91977744b6

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31623963903)
