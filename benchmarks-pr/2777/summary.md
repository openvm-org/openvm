| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/fibonacci-28f15696253db32470b1ef2f256747681740db4c.md) | 1,832 |  4,000,051 |  441 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/keccak-28f15696253db32470b1ef2f256747681740db4c.md) | 13,816 |  14,365,133 |  2,192 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/sha2_bench-28f15696253db32470b1ef2f256747681740db4c.md) | 8,092 |  11,167,961 |  902 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/regex-28f15696253db32470b1ef2f256747681740db4c.md) | 1,520 |  4,090,656 |  355 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/ecrecover-28f15696253db32470b1ef2f256747681740db4c.md) | 610 |  112,210 |  270 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/pairing-28f15696253db32470b1ef2f256747681740db4c.md) | 740 |  592,827 |  266 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2777/kitchen_sink-28f15696253db32470b1ef2f256747681740db4c.md) | 1,881 |  1,979,971 |  409 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/28f15696253db32470b1ef2f256747681740db4c

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25939257524)
