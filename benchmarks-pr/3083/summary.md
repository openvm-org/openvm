| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3083/fibonacci-f2a6c35b54a85c3d667ad0bb62e5be1f4231edc6.md) | 456 |  4,000,051 |  240 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3083/keccak-f2a6c35b54a85c3d667ad0bb62e5be1f4231edc6.md) | 7,260 |  14,365,133 |  1,539 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3083/sha2_bench-f2a6c35b54a85c3d667ad0bb62e5be1f4231edc6.md) | 4,728 |  11,167,961 |  529 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3083/regex-f2a6c35b54a85c3d667ad0bb62e5be1f4231edc6.md) | 661 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3083/ecrecover-f2a6c35b54a85c3d667ad0bb62e5be1f4231edc6.md) | 227 |  112,210 |  185 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3083/pairing-f2a6c35b54a85c3d667ad0bb62e5be1f4231edc6.md) | 297 |  592,827 |  188 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3083/kitchen_sink-f2a6c35b54a85c3d667ad0bb62e5be1f4231edc6.md) | 2,624 |  1,979,971 |  457 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/f2a6c35b54a85c3d667ad0bb62e5be1f4231edc6

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30552437871)
