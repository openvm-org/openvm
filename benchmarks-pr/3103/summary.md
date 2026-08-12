| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/fibonacci-dc19101e912e9ec41f59d4b1d50839d98590c477.md) | 470 |  4,000,051 |  232 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/keccak-dc19101e912e9ec41f59d4b1d50839d98590c477.md) | 7,369 |  14,365,133 |  1,519 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/sha2_bench-dc19101e912e9ec41f59d4b1d50839d98590c477.md) | 4,155 |  11,167,961 |  517 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/regex-dc19101e912e9ec41f59d4b1d50839d98590c477.md) | 667 |  4,090,656 |  213 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/ecrecover-dc19101e912e9ec41f59d4b1d50839d98590c477.md) | 194 |  112,210 |  196 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/pairing-dc19101e912e9ec41f59d4b1d50839d98590c477.md) | 233 |  592,827 |  197 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/kitchen_sink-dc19101e912e9ec41f59d4b1d50839d98590c477.md) | 2,040 |  1,979,971 |  526 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/dc19101e912e9ec41f59d4b1d50839d98590c477

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31608482209)
