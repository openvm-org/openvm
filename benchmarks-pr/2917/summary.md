| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2917/fibonacci-52a3e404114959c2f6c7318f201400cda2091f76.md) | 1,023 |  4,000,051 |  390 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2917/keccak-52a3e404114959c2f6c7318f201400cda2091f76.md) | 16,254 |  14,365,133 |  3,009 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2917/sha2_bench-52a3e404114959c2f6c7318f201400cda2091f76.md) | 8,312 |  11,167,961 |  1,013 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2917/regex-52a3e404114959c2f6c7318f201400cda2091f76.md) | 1,229 |  4,090,656 |  353 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2917/ecrecover-52a3e404114959c2f6c7318f201400cda2091f76.md) | 435 |  112,210 |  284 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2917/pairing-52a3e404114959c2f6c7318f201400cda2091f76.md) | 594 |  592,827 |  294 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2917/kitchen_sink-52a3e404114959c2f6c7318f201400cda2091f76.md) | 3,894 |  1,979,971 |  862 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/52a3e404114959c2f6c7318f201400cda2091f76

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27936601629)
