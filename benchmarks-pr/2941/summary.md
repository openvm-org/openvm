| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2941/fibonacci-15c60480425a2ee4ad6d3b3336b7c73783bc9a6f.md) | 3,090 |  12,000,265 | <span style='color: green'>(-3522 [-83.9%])</span> 677 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2941/keccak-15c60480425a2ee4ad6d3b3336b7c73783bc9a6f.md) | 16,946 |  18,655,329 |  3,149 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2941/sha2_bench-15c60480425a2ee4ad6d3b3336b7c73783bc9a6f.md) | 9,198 |  14,793,960 |  1,125 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2941/regex-15c60480425a2ee4ad6d3b3336b7c73783bc9a6f.md) | 1,170 |  4,137,067 | <span style='color: green'>(-11769 [-97.1%])</span> 352 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2941/ecrecover-15c60480425a2ee4ad6d3b3336b7c73783bc9a6f.md) | 607 |  123,583 | <span style='color: green'>(-5977 [-95.3%])</span> 294 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2941/pairing-15c60480425a2ee4ad6d3b3336b7c73783bc9a6f.md) | 936 |  1,745,757 | <span style='color: green'>(-6345 [-95.4%])</span> 308 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2941/kitchen_sink-15c60480425a2ee4ad6d3b3336b7c73783bc9a6f.md) | 4,109 |  2,579,903 |  884 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2941/fibonacci_e2e-15c60480425a2ee4ad6d3b3336b7c73783bc9a6f.md) | 1,523 |  12,000,265 |  289 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2941/regex_e2e-15c60480425a2ee4ad6d3b3336b7c73783bc9a6f.md) | 770 |  4,137,067 |  166 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2941/ecrecover_e2e-15c60480425a2ee4ad6d3b3336b7c73783bc9a6f.md) | 507 |  123,583 |  142 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2941/pairing_e2e-15c60480425a2ee4ad6d3b3336b7c73783bc9a6f.md) | 654 |  1,745,757 |  147 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2941/kitchen_sink_e2e-15c60480425a2ee4ad6d3b3336b7c73783bc9a6f.md) | 2,334 |  2,579,903 |  387 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/15c60480425a2ee4ad6d3b3336b7c73783bc9a6f

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28557372693)
