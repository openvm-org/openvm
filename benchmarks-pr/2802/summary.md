| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2802/fibonacci-7acbea77de21a4580878a50eec63915b4bf0c383.md) | 1,577 |  4,000,051 |  439 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2802/keccak-7acbea77de21a4580878a50eec63915b4bf0c383.md) | 14,327 |  14,365,133 |  2,254 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2802/sha2_bench-7acbea77de21a4580878a50eec63915b4bf0c383.md) | 9,005 |  11,167,961 |  1,387 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2802/regex-7acbea77de21a4580878a50eec63915b4bf0c383.md) | 1,478 |  4,090,656 |  359 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2802/ecrecover-7acbea77de21a4580878a50eec63915b4bf0c383.md) | 495 |  112,210 |  277 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2802/pairing-7acbea77de21a4580878a50eec63915b4bf0c383.md) | 596 |  592,827 |  255 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2802/kitchen_sink-7acbea77de21a4580878a50eec63915b4bf0c383.md) | 2,223 |  1,979,971 |  410 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/7acbea77de21a4580878a50eec63915b4bf0c383

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26237737723)
