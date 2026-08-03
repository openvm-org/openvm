| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3035/fibonacci-c0be36c7858142536370432d3b4ee9943ed297c7.md) | 471 |  4,000,051 |  229 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3035/keccak-c0be36c7858142536370432d3b4ee9943ed297c7.md) | 7,314 |  14,365,133 |  1,510 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3035/sha2_bench-c0be36c7858142536370432d3b4ee9943ed297c7.md) | 4,101 |  11,167,961 |  516 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3035/regex-c0be36c7858142536370432d3b4ee9943ed297c7.md) | 664 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3035/ecrecover-c0be36c7858142536370432d3b4ee9943ed297c7.md) | 253 |  78,475 |  228 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3035/pairing-c0be36c7858142536370432d3b4ee9943ed297c7.md) | 236 |  592,827 |  196 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3035/kitchen_sink-c0be36c7858142536370432d3b4ee9943ed297c7.md) | 2,391 |  2,341,811 |  571 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/c0be36c7858142536370432d3b4ee9943ed297c7

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30827291067)
