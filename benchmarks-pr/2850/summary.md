| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/fibonacci-f338acfa1017ba1a4e491ea7b7e4f01003d47338.md) | 5,485 |  4,000,051 |  542 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/keccak-f338acfa1017ba1a4e491ea7b7e4f01003d47338.md) | 20,362 |  14,365,133 |  3,046 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/sha2_bench-f338acfa1017ba1a4e491ea7b7e4f01003d47338.md) | 14,371 |  11,167,961 |  1,960 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/regex-f338acfa1017ba1a4e491ea7b7e4f01003d47338.md) | 3,807 |  4,090,656 |  438 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/ecrecover-f338acfa1017ba1a4e491ea7b7e4f01003d47338.md) | 1,975 |  112,210 |  311 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/pairing-f338acfa1017ba1a4e491ea7b7e4f01003d47338.md) | 2,095 |  592,827 |  299 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2850/kitchen_sink-f338acfa1017ba1a4e491ea7b7e4f01003d47338.md) | 5,624 |  1,979,971 |  872 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/f338acfa1017ba1a4e491ea7b7e4f01003d47338

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27361271670)
