| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2806/fibonacci-6a3b2f55c963db508c6446a75b3e86103ba3db41.md) | 3,705 |  12,000,265 |  912 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2806/keccak-6a3b2f55c963db508c6446a75b3e86103ba3db41.md) | 18,611 |  18,655,329 |  3,276 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2806/sha2_bench-6a3b2f55c963db508c6446a75b3e86103ba3db41.md) | 10,141 |  14,793,960 |  1,457 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2806/regex-6a3b2f55c963db508c6446a75b3e86103ba3db41.md) | 1,398 |  4,137,067 |  359 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2806/ecrecover-6a3b2f55c963db508c6446a75b3e86103ba3db41.md) | 597 |  123,583 |  248 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2806/pairing-6a3b2f55c963db508c6446a75b3e86103ba3db41.md) | 896 |  1,745,757 |  268 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2806/kitchen_sink-6a3b2f55c963db508c6446a75b3e86103ba3db41.md) | 1,898 |  2,579,903 |  413 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/6a3b2f55c963db508c6446a75b3e86103ba3db41

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26278479056)
