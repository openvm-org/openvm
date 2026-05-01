| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2766/fibonacci-a9478a78ce64e6b27e03870df31ef5cdaf4a3f10.md) | 3,832 |  12,000,265 |  953 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2766/keccak-a9478a78ce64e6b27e03870df31ef5cdaf4a3f10.md) | 18,842 |  18,655,329 |  3,365 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2766/sha2_bench-a9478a78ce64e6b27e03870df31ef5cdaf4a3f10.md) | 8,958 |  14,793,960 |  1,395 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2766/regex-a9478a78ce64e6b27e03870df31ef5cdaf4a3f10.md) | 1,435 |  4,137,067 |  386 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2766/ecrecover-a9478a78ce64e6b27e03870df31ef5cdaf4a3f10.md) | 633 |  123,583 |  270 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2766/pairing-a9478a78ce64e6b27e03870df31ef5cdaf4a3f10.md) | 891 |  1,745,757 |  281 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2766/kitchen_sink-a9478a78ce64e6b27e03870df31ef5cdaf4a3f10.md) | 2,093 |  2,579,903 |  438 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/a9478a78ce64e6b27e03870df31ef5cdaf4a3f10

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25233192310)
