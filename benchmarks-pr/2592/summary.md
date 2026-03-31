| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-257e12e9e018a9b5b761be86371bc915bc357258.md) | 3,868 |  12,000,265 |  950 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-257e12e9e018a9b5b761be86371bc915bc357258.md) | 18,394 |  18,655,329 |  3,276 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-257e12e9e018a9b5b761be86371bc915bc357258.md) | 1,440 |  4,137,067 |  378 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-257e12e9e018a9b5b761be86371bc915bc357258.md) | 642 |  123,583 |  267 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-257e12e9e018a9b5b761be86371bc915bc357258.md) | 902 |  1,745,757 |  283 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-257e12e9e018a9b5b761be86371bc915bc357258.md) | 2,288 |  2,579,903 |  442 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/257e12e9e018a9b5b761be86371bc915bc357258

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23818296882)
