| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/fibonacci-240586c895874f9ac7d3bb99f4f1fc3e117ef6ea.md) | 476 |  4,000,051 |  233 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/keccak-240586c895874f9ac7d3bb99f4f1fc3e117ef6ea.md) | 8,741 |  14,365,133 |  1,527 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/sha2_bench-240586c895874f9ac7d3bb99f4f1fc3e117ef6ea.md) | 3,926 |  11,167,961 |  521 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/regex-240586c895874f9ac7d3bb99f4f1fc3e117ef6ea.md) | 504 |  4,090,656 |  189 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/ecrecover-240586c895874f9ac7d3bb99f4f1fc3e117ef6ea.md) | 217 |  112,210 |  180 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/pairing-240586c895874f9ac7d3bb99f4f1fc3e117ef6ea.md) | 265 |  592,827 |  182 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/kitchen_sink-240586c895874f9ac7d3bb99f4f1fc3e117ef6ea.md) | 1,925 |  1,979,971 |  460 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/fibonacci_e2e-240586c895874f9ac7d3bb99f4f1fc3e117ef6ea.md) | 503 |  4,000,051 |  220 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/regex_e2e-240586c895874f9ac7d3bb99f4f1fc3e117ef6ea.md) | 589 |  4,090,656 |  181 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/ecrecover_e2e-240586c895874f9ac7d3bb99f4f1fc3e117ef6ea.md) | 217 |  112,210 |  178 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/pairing_e2e-240586c895874f9ac7d3bb99f4f1fc3e117ef6ea.md) | 281 |  592,827 |  172 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/kitchen_sink_e2e-240586c895874f9ac7d3bb99f4f1fc3e117ef6ea.md) | 2,255 |  1,979,971 |  452 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/240586c895874f9ac7d3bb99f4f1fc3e117ef6ea

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29401087300)
