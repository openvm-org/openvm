| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2954/fibonacci-20f1ac38b06487b1375a42e8359d903a615d628c.md) | 3,063 |  12,000,265 |  675 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2954/keccak-20f1ac38b06487b1375a42e8359d903a615d628c.md) | 16,386 |  18,655,329 |  3,054 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2954/sha2_bench-20f1ac38b06487b1375a42e8359d903a615d628c.md) | 9,121 |  14,793,960 |  1,120 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2954/regex-20f1ac38b06487b1375a42e8359d903a615d628c.md) | 1,179 |  4,137,067 |  357 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2954/ecrecover-20f1ac38b06487b1375a42e8359d903a615d628c.md) | 615 |  123,583 |  292 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2954/pairing-20f1ac38b06487b1375a42e8359d903a615d628c.md) | 941 |  1,745,757 |  307 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2954/kitchen_sink-20f1ac38b06487b1375a42e8359d903a615d628c.md) | 4,146 |  2,579,903 |  888 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/20f1ac38b06487b1375a42e8359d903a615d628c

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28482889991)
