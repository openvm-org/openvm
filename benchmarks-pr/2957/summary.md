| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2957/fibonacci-bd7eb1b2937f97df326ed49da3263ee368c815c3.md) | 884 |  4,000,051 |  390 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2957/keccak-bd7eb1b2937f97df326ed49da3263ee368c815c3.md) | 15,766 |  14,365,133 |  3,070 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2957/sha2_bench-bd7eb1b2937f97df326ed49da3263ee368c815c3.md) | 8,111 |  11,167,961 |  994 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2957/regex-bd7eb1b2937f97df326ed49da3263ee368c815c3.md) | 1,190 |  4,090,656 |  354 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2957/ecrecover-bd7eb1b2937f97df326ed49da3263ee368c815c3.md) | 435 |  112,210 |  281 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2957/pairing-bd7eb1b2937f97df326ed49da3263ee368c815c3.md) | 576 |  592,827 |  297 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2957/kitchen_sink-bd7eb1b2937f97df326ed49da3263ee368c815c3.md) | 3,834 |  1,979,971 |  864 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/bd7eb1b2937f97df326ed49da3263ee368c815c3

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28962287219)
