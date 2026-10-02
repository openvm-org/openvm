| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3163/fibonacci-a2065ffbc769bd376d6c79d8ea6d8657bc748f51.md) | 474 |  4,000,051 |  234 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3163/keccak-a2065ffbc769bd376d6c79d8ea6d8657bc748f51.md) | 7,762 |  14,365,133 |  1,565 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3163/sha2_bench-a2065ffbc769bd376d6c79d8ea6d8657bc748f51.md) | 4,246 |  11,167,961 |  529 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3163/regex-a2065ffbc769bd376d6c79d8ea6d8657bc748f51.md) | 759 |  4,090,656 |  217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3163/ecrecover-a2065ffbc769bd376d6c79d8ea6d8657bc748f51.md) | 215 |  112,210 |  186 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3163/pairing-a2065ffbc769bd376d6c79d8ea6d8657bc748f51.md) | 254 |  592,827 |  186 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3163/kitchen_sink-a2065ffbc769bd376d6c79d8ea6d8657bc748f51.md) | 2,247 |  1,979,971 |  472 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/a2065ffbc769bd376d6c79d8ea6d8657bc748f51

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/37038981023)
