| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/fibonacci-fb8599276b9de585910b883aa04d35b54b7234b8.md) | 416 |  4,000,051 |  235 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/keccak-fb8599276b9de585910b883aa04d35b54b7234b8.md) | 8,662 |  14,365,133 |  1,530 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/sha2_bench-fb8599276b9de585910b883aa04d35b54b7234b8.md) | 4,235 |  11,167,961 |  527 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/regex-fb8599276b9de585910b883aa04d35b54b7234b8.md) | 576 |  4,090,656 |  219 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/ecrecover-fb8599276b9de585910b883aa04d35b54b7234b8.md) | 219 |  112,210 |  182 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/pairing-fb8599276b9de585910b883aa04d35b54b7234b8.md) | 293 |  592,827 |  188 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3040/kitchen_sink-fb8599276b9de585910b883aa04d35b54b7234b8.md) | 1,914 |  1,979,971 |  465 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/fb8599276b9de585910b883aa04d35b54b7234b8

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29763669030)
