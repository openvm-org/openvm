| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2807/fibonacci-c345409c7fea02c117b78ce45afc3296853080e0.md) | 3,730 |  12,000,265 |  913 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2807/keccak-c345409c7fea02c117b78ce45afc3296853080e0.md) | 18,835 |  18,655,329 |  3,324 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2807/sha2_bench-c345409c7fea02c117b78ce45afc3296853080e0.md) | 10,367 |  14,793,960 |  1,480 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2807/regex-c345409c7fea02c117b78ce45afc3296853080e0.md) | 1,397 |  4,137,067 |  353 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2807/ecrecover-c345409c7fea02c117b78ce45afc3296853080e0.md) | 594 |  123,583 |  249 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2807/pairing-c345409c7fea02c117b78ce45afc3296853080e0.md) | 884 |  1,745,757 |  262 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2807/kitchen_sink-c345409c7fea02c117b78ce45afc3296853080e0.md) | 1,901 |  2,579,903 |  411 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/c345409c7fea02c117b78ce45afc3296853080e0

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26286746606)
