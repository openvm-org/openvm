| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3102/fibonacci-50fcc3cec3d47c0ec2590b62f59dfb1e72b7bb19.md) | 476 |  4,000,051 |  229 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3102/keccak-50fcc3cec3d47c0ec2590b62f59dfb1e72b7bb19.md) | 7,368 |  14,365,133 |  1,522 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3102/sha2_bench-50fcc3cec3d47c0ec2590b62f59dfb1e72b7bb19.md) | 4,142 |  11,167,961 |  533 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3102/regex-50fcc3cec3d47c0ec2590b62f59dfb1e72b7bb19.md) | 660 |  4,090,656 |  213 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3102/ecrecover-50fcc3cec3d47c0ec2590b62f59dfb1e72b7bb19.md) | 223 |  112,210 |  181 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3102/pairing-50fcc3cec3d47c0ec2590b62f59dfb1e72b7bb19.md) | 234 |  592,827 |  182 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3102/kitchen_sink-50fcc3cec3d47c0ec2590b62f59dfb1e72b7bb19.md) | 2,028 |  1,979,971 |  458 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/50fcc3cec3d47c0ec2590b62f59dfb1e72b7bb19

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31083338103)
