| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/fibonacci-1942cd5e89fc0297b2e3757d34c04107dbb47c3f.md) | 424 |  4,000,051 |  228 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/keccak-1942cd5e89fc0297b2e3757d34c04107dbb47c3f.md) | 8,453 |  14,365,133 |  1,520 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/sha2_bench-1942cd5e89fc0297b2e3757d34c04107dbb47c3f.md) | 4,172 |  11,167,961 |  527 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/regex-1942cd5e89fc0297b2e3757d34c04107dbb47c3f.md) | 508 |  4,090,656 |  191 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/ecrecover-1942cd5e89fc0297b2e3757d34c04107dbb47c3f.md) | 227 |  112,210 |  185 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/pairing-1942cd5e89fc0297b2e3757d34c04107dbb47c3f.md) | 268 |  592,827 |  186 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3023/kitchen_sink-1942cd5e89fc0297b2e3757d34c04107dbb47c3f.md) | 1,885 |  1,979,971 |  459 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/1942cd5e89fc0297b2e3757d34c04107dbb47c3f

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29493316362)
