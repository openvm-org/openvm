| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/fibonacci-d642709f3ab403a9c841dca6adb2fc0cf2c8f13d.md) | 1,398 |  4,000,051 |  429 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/keccak-d642709f3ab403a9c841dca6adb2fc0cf2c8f13d.md) | 13,223 |  14,365,133 |  2,193 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/sha2_bench-d642709f3ab403a9c841dca6adb2fc0cf2c8f13d.md) | 8,937 |  11,167,961 |  1,407 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/regex-d642709f3ab403a9c841dca6adb2fc0cf2c8f13d.md) | 1,357 |  4,090,656 |  360 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/ecrecover-d642709f3ab403a9c841dca6adb2fc0cf2c8f13d.md) | 467 |  112,210 |  266 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/pairing-d642709f3ab403a9c841dca6adb2fc0cf2c8f13d.md) | 590 |  592,827 |  257 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2778/kitchen_sink-d642709f3ab403a9c841dca6adb2fc0cf2c8f13d.md) | 2,200 |  1,979,971 |  414 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/d642709f3ab403a9c841dca6adb2fc0cf2c8f13d

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25968676574)
