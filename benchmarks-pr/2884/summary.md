| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2884/fibonacci-a8a5a80622094316ec04564cc4ac6cfda64f6bc1.md) | 3,926 |  12,000,265 |  1,134 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2884/keccak-a8a5a80622094316ec04564cc4ac6cfda64f6bc1.md) | 21,792 |  18,655,329 |  4,619 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2884/sha2_bench-a8a5a80622094316ec04564cc4ac6cfda64f6bc1.md) | 9,595 |  14,793,960 |  1,852 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2884/regex-a8a5a80622094316ec04564cc4ac6cfda64f6bc1.md) | 1,496 |  4,137,067 |  429 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2884/ecrecover-a8a5a80622094316ec04564cc4ac6cfda64f6bc1.md) | 610 |  123,583 |  288 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2884/pairing-a8a5a80622094316ec04564cc4ac6cfda64f6bc1.md) | 939 |  1,745,757 |  306 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2884/kitchen_sink-a8a5a80622094316ec04564cc4ac6cfda64f6bc1.md) | 4,175 |  2,579,903 |  888 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/a8a5a80622094316ec04564cc4ac6cfda64f6bc1

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27444567712)
