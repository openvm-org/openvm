| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/fibonacci-3c69405ade3af8830807d92ee2b694fbf030c841.md) | 473 |  4,000,051 |  243 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/keccak-3c69405ade3af8830807d92ee2b694fbf030c841.md) | 9,842 |  14,365,133 |  1,522 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/sha2_bench-3c69405ade3af8830807d92ee2b694fbf030c841.md) | 4,790 |  11,167,961 |  536 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/regex-3c69405ade3af8830807d92ee2b694fbf030c841.md) | 680 |  4,090,656 |  220 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/ecrecover-3c69405ade3af8830807d92ee2b694fbf030c841.md) | 223 |  112,210 |  185 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/pairing-3c69405ade3af8830807d92ee2b694fbf030c841.md) | 268 |  592,827 |  187 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/kitchen_sink-3c69405ade3af8830807d92ee2b694fbf030c841.md) | 2,822 |  1,979,971 |  469 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/3c69405ade3af8830807d92ee2b694fbf030c841

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30481385694)
