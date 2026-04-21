| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-c40a8434c4c9551c801898bd4882154fe453c97d.md) | 3,818 |  12,000,265 |  944 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-c40a8434c4c9551c801898bd4882154fe453c97d.md) | 18,747 |  18,655,329 |  3,343 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/sha2_bench-c40a8434c4c9551c801898bd4882154fe453c97d.md) | 8,965 |  14,793,960 |  1,391 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-c40a8434c4c9551c801898bd4882154fe453c97d.md) | 1,431 |  4,137,067 |  380 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-c40a8434c4c9551c801898bd4882154fe453c97d.md) | 644 |  123,583 |  273 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-c40a8434c4c9551c801898bd4882154fe453c97d.md) | 919 |  1,745,757 |  287 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-c40a8434c4c9551c801898bd4882154fe453c97d.md) | 2,094 |  2,579,903 |  439 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci_e2e-c40a8434c4c9551c801898bd4882154fe453c97d.md) | 1,865 |  12,000,265 |  459 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex_e2e-c40a8434c4c9551c801898bd4882154fe453c97d.md) | 852 |  4,137,067 |  192 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover_e2e-c40a8434c4c9551c801898bd4882154fe453c97d.md) | 553 |  123,583 |  152 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing_e2e-c40a8434c4c9551c801898bd4882154fe453c97d.md) | 660 |  1,745,757 |  153 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink_e2e-c40a8434c4c9551c801898bd4882154fe453c97d.md) | 2,203 |  2,579,903 |  424 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/c40a8434c4c9551c801898bd4882154fe453c97d

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24732399960)
