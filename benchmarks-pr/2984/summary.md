| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/fibonacci-280289c39e523f47e76503e0db170de9e70d8e44.md) | 469 |  4,000,051 |  242 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/keccak-280289c39e523f47e76503e0db170de9e70d8e44.md) | 7,316 |  14,365,133 |  1,537 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/sha2_bench-280289c39e523f47e76503e0db170de9e70d8e44.md) | 4,732 |  11,167,961 |  529 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/regex-280289c39e523f47e76503e0db170de9e70d8e44.md) | 666 |  4,090,656 |  219 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/ecrecover-280289c39e523f47e76503e0db170de9e70d8e44.md) | 227 |  112,210 |  186 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/pairing-280289c39e523f47e76503e0db170de9e70d8e44.md) | 269 |  592,827 |  189 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/kitchen_sink-280289c39e523f47e76503e0db170de9e70d8e44.md) | 2,724 |  1,979,971 |  471 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/280289c39e523f47e76503e0db170de9e70d8e44

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30133966678)
