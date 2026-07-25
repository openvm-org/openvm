| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/fibonacci-abd0e3af607259fe12f097b2c548bffa9829880c.md) | 479 |  4,000,051 |  243 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/keccak-abd0e3af607259fe12f097b2c548bffa9829880c.md) | 10,288 |  14,365,133 |  1,534 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/sha2_bench-abd0e3af607259fe12f097b2c548bffa9829880c.md) | 4,668 |  11,167,961 |  528 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/regex-abd0e3af607259fe12f097b2c548bffa9829880c.md) | 681 |  4,090,656 |  214 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/ecrecover-abd0e3af607259fe12f097b2c548bffa9829880c.md) | 228 |  112,210 |  188 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/pairing-abd0e3af607259fe12f097b2c548bffa9829880c.md) | 275 |  592,827 |  188 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3020/kitchen_sink-abd0e3af607259fe12f097b2c548bffa9829880c.md) | 2,372 |  1,979,971 |  461 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/abd0e3af607259fe12f097b2c548bffa9829880c

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30145560493)
