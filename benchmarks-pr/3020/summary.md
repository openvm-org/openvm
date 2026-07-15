| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/fibonacci-3b26b3dcf797d98f2a73cf4f9b27d192ad235511.md) | 466 |  4,000,051 |  239 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/keccak-3b26b3dcf797d98f2a73cf4f9b27d192ad235511.md) | 7,096 |  14,365,133 |  1,528 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/sha2_bench-3b26b3dcf797d98f2a73cf4f9b27d192ad235511.md) | 4,402 |  11,167,961 |  529 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/regex-3b26b3dcf797d98f2a73cf4f9b27d192ad235511.md) | 676 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/ecrecover-3b26b3dcf797d98f2a73cf4f9b27d192ad235511.md) | 222 |  112,210 |  190 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/pairing-3b26b3dcf797d98f2a73cf4f9b27d192ad235511.md) | 250 |  592,827 |  185 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/kitchen_sink-3b26b3dcf797d98f2a73cf4f9b27d192ad235511.md) | 2,720 |  1,979,971 |  477 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/3b26b3dcf797d98f2a73cf4f9b27d192ad235511

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29437763707)
