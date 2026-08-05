| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3110/fibonacci-d2af96540085aba258d78f835588d5f9e61dcb79.md) | 480 |  4,000,051 |  231 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3110/keccak-d2af96540085aba258d78f835588d5f9e61dcb79.md) | 7,353 |  14,365,133 |  1,526 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3110/sha2_bench-d2af96540085aba258d78f835588d5f9e61dcb79.md) | 4,167 |  11,167,961 |  525 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3110/regex-d2af96540085aba258d78f835588d5f9e61dcb79.md) | 662 |  4,090,656 |  214 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3110/ecrecover-d2af96540085aba258d78f835588d5f9e61dcb79.md) | 222 |  112,210 |  181 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3110/pairing-d2af96540085aba258d78f835588d5f9e61dcb79.md) | 236 |  592,827 |  184 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3110/kitchen_sink-d2af96540085aba258d78f835588d5f9e61dcb79.md) | 2,045 |  1,979,971 |  463 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/d2af96540085aba258d78f835588d5f9e61dcb79

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31039140576)
