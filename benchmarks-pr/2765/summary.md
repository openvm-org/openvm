| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/fibonacci-035b91e6974b8c530f3495fe403731578e66085b.md) | 1,901 |  4,000,051 |  543 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/keccak-035b91e6974b8c530f3495fe403731578e66085b.md) | 13,472 |  14,365,133 |  2,223 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/sha2_bench-035b91e6974b8c530f3495fe403731578e66085b.md) | 9,421 |  11,167,961 |  1,402 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/regex-035b91e6974b8c530f3495fe403731578e66085b.md) | 1,600 |  4,090,656 |  380 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/ecrecover-035b91e6974b8c530f3495fe403731578e66085b.md) | 642 |  112,210 |  291 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/pairing-035b91e6974b8c530f3495fe403731578e66085b.md) | 762 |  592,827 |  274 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/kitchen_sink-035b91e6974b8c530f3495fe403731578e66085b.md) | 2,038 |  1,979,971 |  435 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/035b91e6974b8c530f3495fe403731578e66085b

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25852568485)
