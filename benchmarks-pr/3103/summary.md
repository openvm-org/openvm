| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/fibonacci-325eb810185d53d14ff6fd7fc0bfb64fb6b0701e.md) | 475 |  4,000,051 |  230 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/keccak-325eb810185d53d14ff6fd7fc0bfb64fb6b0701e.md) | 7,289 |  14,365,133 |  1,520 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/sha2_bench-325eb810185d53d14ff6fd7fc0bfb64fb6b0701e.md) | 4,129 |  11,167,961 |  525 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/regex-325eb810185d53d14ff6fd7fc0bfb64fb6b0701e.md) | 660 |  4,090,656 |  212 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/ecrecover-325eb810185d53d14ff6fd7fc0bfb64fb6b0701e.md) | 222 |  112,210 |  193 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/pairing-325eb810185d53d14ff6fd7fc0bfb64fb6b0701e.md) | 231 |  592,827 |  196 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3103/kitchen_sink-325eb810185d53d14ff6fd7fc0bfb64fb6b0701e.md) | 2,031 |  1,979,971 |  523 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/325eb810185d53d14ff6fd7fc0bfb64fb6b0701e

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31418144917)
