| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3080/fibonacci-acd7efa4fa8b8599b375e865c80df3bdf4f78227.md) | 467 |  4,000,051 |  240 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3080/keccak-acd7efa4fa8b8599b375e865c80df3bdf4f78227.md) | 7,365 |  14,365,133 |  1,553 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3080/sha2_bench-acd7efa4fa8b8599b375e865c80df3bdf4f78227.md) | 4,750 |  11,167,961 |  529 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3080/regex-acd7efa4fa8b8599b375e865c80df3bdf4f78227.md) | 671 |  4,090,656 |  217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3080/ecrecover-acd7efa4fa8b8599b375e865c80df3bdf4f78227.md) | 233 |  112,210 |  187 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3080/pairing-acd7efa4fa8b8599b375e865c80df3bdf4f78227.md) | 320 |  592,827 |  188 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3080/kitchen_sink-acd7efa4fa8b8599b375e865c80df3bdf4f78227.md) | 2,671 |  1,979,971 |  472 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/acd7efa4fa8b8599b375e865c80df3bdf4f78227

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30371866254)
