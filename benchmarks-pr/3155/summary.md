| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3155/fibonacci-20250c3c21214f516d908713ee5181393444f644.md) | 486 |  4,000,051 |  235 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3155/keccak-20250c3c21214f516d908713ee5181393444f644.md) | 7,617 |  14,365,133 |  1,622 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3155/sha2_bench-20250c3c21214f516d908713ee5181393444f644.md) | 4,333 |  11,167,961 |  525 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3155/regex-20250c3c21214f516d908713ee5181393444f644.md) | 766 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3155/ecrecover-20250c3c21214f516d908713ee5181393444f644.md) | 210 |  112,210 |  188 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3155/pairing-20250c3c21214f516d908713ee5181393444f644.md) | 249 |  592,827 |  172 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3155/kitchen_sink-20250c3c21214f516d908713ee5181393444f644.md) | 2,263 |  1,979,971 |  477 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/20250c3c21214f516d908713ee5181393444f644

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/34457040220)
