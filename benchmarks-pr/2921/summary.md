| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2921/fibonacci-810d54a93d26745e2480354291ae425fb4a7d28d.md) | 1,025 |  4,000,051 |  389 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2921/keccak-810d54a93d26745e2480354291ae425fb4a7d28d.md) | 15,811 |  14,365,133 |  3,039 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2921/sha2_bench-810d54a93d26745e2480354291ae425fb4a7d28d.md) | 8,147 |  11,167,961 |  1,001 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2921/regex-810d54a93d26745e2480354291ae425fb4a7d28d.md) | 1,234 |  4,090,656 |  375 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2921/ecrecover-810d54a93d26745e2480354291ae425fb4a7d28d.md) | 434 |  112,210 |  284 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2921/pairing-810d54a93d26745e2480354291ae425fb4a7d28d.md) | 601 |  592,827 |  305 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2921/kitchen_sink-810d54a93d26745e2480354291ae425fb4a7d28d.md) | 3,883 |  1,979,971 |  864 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/810d54a93d26745e2480354291ae425fb4a7d28d

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28059621889)
