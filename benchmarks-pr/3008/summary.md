| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3008/fibonacci-8b0a61a9088e5817f0bbfbbbe65fb52895f34831.md) | 933 |  4,000,051 |  396 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3008/keccak-8b0a61a9088e5817f0bbfbbbe65fb52895f34831.md) | 14,842 |  14,365,133 |  3,020 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3008/sha2_bench-8b0a61a9088e5817f0bbfbbbe65fb52895f34831.md) | 8,470 |  11,167,961 |  1,009 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3008/regex-8b0a61a9088e5817f0bbfbbbe65fb52895f34831.md) | 1,117 |  4,090,656 |  359 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3008/ecrecover-8b0a61a9088e5817f0bbfbbbe65fb52895f34831.md) | 316 |  112,210 |  273 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3008/pairing-8b0a61a9088e5817f0bbfbbbe65fb52895f34831.md) | 474 |  592,827 |  301 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3008/kitchen_sink-8b0a61a9088e5817f0bbfbbbe65fb52895f34831.md) | 4,181 |  1,979,971 |  862 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/8b0a61a9088e5817f0bbfbbbe65fb52895f34831

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29272219520)
