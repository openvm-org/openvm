| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2807/fibonacci-d133a83ae9b79697c687688342359bb75c9800fd.md) | 3,760 |  12,000,265 |  923 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2807/keccak-d133a83ae9b79697c687688342359bb75c9800fd.md) | 18,324 |  18,655,329 |  3,238 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2807/sha2_bench-d133a83ae9b79697c687688342359bb75c9800fd.md) | 10,398 |  14,793,960 |  1,486 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2807/regex-d133a83ae9b79697c687688342359bb75c9800fd.md) | 1,415 |  4,137,067 |  360 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2807/ecrecover-d133a83ae9b79697c687688342359bb75c9800fd.md) | 611 |  123,583 |  250 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2807/pairing-d133a83ae9b79697c687688342359bb75c9800fd.md) | 892 |  1,745,757 |  270 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2807/kitchen_sink-d133a83ae9b79697c687688342359bb75c9800fd.md) | 1,896 |  2,579,903 |  416 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/d133a83ae9b79697c687688342359bb75c9800fd

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26287781972)
