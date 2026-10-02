| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3164/fibonacci-b63fae7e35e581477a60a7e166cd469b7b1124de.md) | 469 |  4,000,051 |  231 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3164/keccak-b63fae7e35e581477a60a7e166cd469b7b1124de.md) | 7,672 |  14,365,133 |  1,655 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3164/sha2_bench-b63fae7e35e581477a60a7e166cd469b7b1124de.md) | 4,256 |  11,167,961 |  520 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3164/regex-b63fae7e35e581477a60a7e166cd469b7b1124de.md) | 738 |  4,090,656 |  214 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3164/ecrecover-b63fae7e35e581477a60a7e166cd469b7b1124de.md) | 206 |  112,210 |  188 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3164/pairing-b63fae7e35e581477a60a7e166cd469b7b1124de.md) | 253 |  592,827 |  173 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3164/kitchen_sink-b63fae7e35e581477a60a7e166cd469b7b1124de.md) | 2,261 |  1,979,971 |  477 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/b63fae7e35e581477a60a7e166cd469b7b1124de

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/37041488661)
