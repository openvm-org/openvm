| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/fibonacci-0f85ced8e1ae2aacf949d3e90403d2675f07f1bb.md) | 413 |  4,000,051 |  226 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/keccak-0f85ced8e1ae2aacf949d3e90403d2675f07f1bb.md) | 8,470 |  14,365,133 |  1,526 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/sha2_bench-0f85ced8e1ae2aacf949d3e90403d2675f07f1bb.md) | 4,127 |  11,167,961 |  523 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/regex-0f85ced8e1ae2aacf949d3e90403d2675f07f1bb.md) | 500 |  4,090,656 |  189 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/ecrecover-0f85ced8e1ae2aacf949d3e90403d2675f07f1bb.md) | 220 |  112,210 |  182 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/pairing-0f85ced8e1ae2aacf949d3e90403d2675f07f1bb.md) | 264 |  592,827 |  183 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/kitchen_sink-0f85ced8e1ae2aacf949d3e90403d2675f07f1bb.md) | 1,883 |  1,979,971 |  460 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/0f85ced8e1ae2aacf949d3e90403d2675f07f1bb

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29435532203)
