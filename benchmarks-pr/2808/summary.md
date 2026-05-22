| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/fibonacci-caee7ffff25d219a7007373d1d6919a485dcdbc7.md) | 1,579 |  4,000,051 |  435 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/keccak-caee7ffff25d219a7007373d1d6919a485dcdbc7.md) | 13,834 |  14,365,133 |  2,356 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/sha2_bench-caee7ffff25d219a7007373d1d6919a485dcdbc7.md) | 9,183 |  11,167,961 |  1,402 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/regex-caee7ffff25d219a7007373d1d6919a485dcdbc7.md) | 1,474 |  4,090,656 |  356 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/ecrecover-caee7ffff25d219a7007373d1d6919a485dcdbc7.md) | 474 |  112,210 |  269 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/pairing-caee7ffff25d219a7007373d1d6919a485dcdbc7.md) | 592 |  592,827 |  255 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2808/kitchen_sink-caee7ffff25d219a7007373d1d6919a485dcdbc7.md) | 1,809 |  1,979,971 |  403 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/caee7ffff25d219a7007373d1d6919a485dcdbc7

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26295045162)
