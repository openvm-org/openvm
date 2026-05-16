| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2786/fibonacci-ecf194e751c3eec1422578778af2c987133a0848.md) | 3,764 |  12,000,265 |  913 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2786/keccak-ecf194e751c3eec1422578778af2c987133a0848.md) | 18,382 |  18,655,329 |  3,250 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2786/sha2_bench-ecf194e751c3eec1422578778af2c987133a0848.md) | 10,277 |  14,793,960 |  1,480 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2786/regex-ecf194e751c3eec1422578778af2c987133a0848.md) | 1,399 |  4,137,067 |  356 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2786/ecrecover-ecf194e751c3eec1422578778af2c987133a0848.md) | 594 |  123,583 |  251 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2786/pairing-ecf194e751c3eec1422578778af2c987133a0848.md) | 887 |  1,745,757 |  267 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2786/kitchen_sink-ecf194e751c3eec1422578778af2c987133a0848.md) | 1,899 |  2,579,903 |  413 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/ecf194e751c3eec1422578778af2c987133a0848

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25947054922)
