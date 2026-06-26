| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/fibonacci-39d54cbff328156b3f2a12aa5140e9aac9096d2d.md) | 3,074 |  12,000,265 |  684 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/keccak-39d54cbff328156b3f2a12aa5140e9aac9096d2d.md) | 16,498 |  18,655,329 |  3,070 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/sha2_bench-39d54cbff328156b3f2a12aa5140e9aac9096d2d.md) | 9,205 |  14,793,960 |  1,125 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/regex-39d54cbff328156b3f2a12aa5140e9aac9096d2d.md) | 1,173 |  4,137,067 |  356 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/ecrecover-39d54cbff328156b3f2a12aa5140e9aac9096d2d.md) | 599 |  123,583 |  285 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/pairing-39d54cbff328156b3f2a12aa5140e9aac9096d2d.md) | 942 |  1,745,757 |  310 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/kitchen_sink-39d54cbff328156b3f2a12aa5140e9aac9096d2d.md) | 4,116 |  2,579,903 |  877 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/39d54cbff328156b3f2a12aa5140e9aac9096d2d

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28267314721)
