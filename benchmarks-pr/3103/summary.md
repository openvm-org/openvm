| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/fibonacci-c60207d31d8592c767cb8fb32f85aa54641a1861.md) | 465 |  4,000,051 |  231 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/keccak-c60207d31d8592c767cb8fb32f85aa54641a1861.md) | 7,393 |  14,365,133 |  1,529 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/sha2_bench-c60207d31d8592c767cb8fb32f85aa54641a1861.md) | 4,126 |  11,167,961 |  515 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/regex-c60207d31d8592c767cb8fb32f85aa54641a1861.md) | 672 |  4,090,656 |  217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/ecrecover-c60207d31d8592c767cb8fb32f85aa54641a1861.md) | 203 |  112,210 |  200 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/pairing-c60207d31d8592c767cb8fb32f85aa54641a1861.md) | 240 |  592,827 |  197 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/kitchen_sink-c60207d31d8592c767cb8fb32f85aa54641a1861.md) | 2,030 |  1,979,971 |  526 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/c60207d31d8592c767cb8fb32f85aa54641a1861

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31673810686)
