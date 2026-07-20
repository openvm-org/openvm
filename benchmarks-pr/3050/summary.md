| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3050/fibonacci-b68482ca58f94b14950c8a6f6f8a4e2be092d8a5.md) | 410 |  4,000,051 |  233 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3050/keccak-b68482ca58f94b14950c8a6f6f8a4e2be092d8a5.md) | 8,630 |  14,365,133 |  1,528 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3050/sha2_bench-b68482ca58f94b14950c8a6f6f8a4e2be092d8a5.md) | 4,208 |  11,167,961 |  518 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3050/regex-b68482ca58f94b14950c8a6f6f8a4e2be092d8a5.md) | 575 |  4,090,656 |  217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3050/ecrecover-b68482ca58f94b14950c8a6f6f8a4e2be092d8a5.md) | 216 |  112,210 |  182 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3050/pairing-b68482ca58f94b14950c8a6f6f8a4e2be092d8a5.md) | 284 |  592,827 |  184 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3050/kitchen_sink-b68482ca58f94b14950c8a6f6f8a4e2be092d8a5.md) | 1,912 |  1,979,971 |  461 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/b68482ca58f94b14950c8a6f6f8a4e2be092d8a5

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29710324886)
