| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/fibonacci-64ac93e453c8f27aa8700fc67bb90efba6852c77.md) | 464 |  4,000,051 |  235 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/keccak-64ac93e453c8f27aa8700fc67bb90efba6852c77.md) | 8,708 |  14,365,133 |  1,518 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/sha2_bench-64ac93e453c8f27aa8700fc67bb90efba6852c77.md) | 3,952 |  11,167,961 |  524 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/regex-64ac93e453c8f27aa8700fc67bb90efba6852c77.md) | 499 |  4,090,656 |  187 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/ecrecover-64ac93e453c8f27aa8700fc67bb90efba6852c77.md) | 216 |  112,210 |  182 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/pairing-64ac93e453c8f27aa8700fc67bb90efba6852c77.md) | 267 |  592,827 |  183 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/kitchen_sink-64ac93e453c8f27aa8700fc67bb90efba6852c77.md) | 1,925 |  1,979,971 |  472 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/fibonacci_e2e-64ac93e453c8f27aa8700fc67bb90efba6852c77.md) | 487 |  4,000,051 |  217 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/regex_e2e-64ac93e453c8f27aa8700fc67bb90efba6852c77.md) | 586 |  4,090,656 |  181 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/ecrecover_e2e-64ac93e453c8f27aa8700fc67bb90efba6852c77.md) | 218 |  112,210 |  174 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/pairing_e2e-64ac93e453c8f27aa8700fc67bb90efba6852c77.md) | 282 |  592,827 |  175 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/kitchen_sink_e2e-64ac93e453c8f27aa8700fc67bb90efba6852c77.md) | 2,263 |  1,979,971 |  452 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/64ac93e453c8f27aa8700fc67bb90efba6852c77

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29444413391)
