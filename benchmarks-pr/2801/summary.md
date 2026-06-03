| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2801/fibonacci-c593fa674168a3daffffff4181638fe3753ac9c3.md) | 1,558 |  4,000,051 |  436 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2801/keccak-c593fa674168a3daffffff4181638fe3753ac9c3.md) | 13,956 |  14,365,133 |  2,377 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2801/sha2_bench-c593fa674168a3daffffff4181638fe3753ac9c3.md) | 9,067 |  11,167,961 |  1,395 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2801/regex-c593fa674168a3daffffff4181638fe3753ac9c3.md) | 1,591 |  4,090,656 |  359 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2801/ecrecover-c593fa674168a3daffffff4181638fe3753ac9c3.md) | 485 |  112,210 |  271 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2801/pairing-c593fa674168a3daffffff4181638fe3753ac9c3.md) | 602 |  592,827 |  252 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2801/kitchen_sink-c593fa674168a3daffffff4181638fe3753ac9c3.md) | 2,002 |  1,979,971 |  408 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/c593fa674168a3daffffff4181638fe3753ac9c3

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26889142669)
