| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2713/fibonacci-d4d2ec8fc5f6fbd1aea751d06892d2accaa49ec0.md) | 3,889 |  12,000,265 |  961 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2713/keccak-d4d2ec8fc5f6fbd1aea751d06892d2accaa49ec0.md) | 18,638 |  18,655,329 |  3,311 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2713/sha2_bench-d4d2ec8fc5f6fbd1aea751d06892d2accaa49ec0.md) | 9,093 |  14,793,960 |  1,409 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2713/regex-d4d2ec8fc5f6fbd1aea751d06892d2accaa49ec0.md) | 1,417 |  4,137,067 |  373 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2713/ecrecover-d4d2ec8fc5f6fbd1aea751d06892d2accaa49ec0.md) | 646 |  123,583 |  271 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2713/pairing-d4d2ec8fc5f6fbd1aea751d06892d2accaa49ec0.md) | 898 |  1,745,757 |  282 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2713/kitchen_sink-d4d2ec8fc5f6fbd1aea751d06892d2accaa49ec0.md) | 2,082 |  2,579,903 |  434 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/d4d2ec8fc5f6fbd1aea751d06892d2accaa49ec0

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24580860584)
