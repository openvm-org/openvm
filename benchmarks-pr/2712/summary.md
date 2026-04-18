| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/fibonacci-914fbe99c8fbf8fe79ceba9c4b6e84b76930bdc5.md) | 3,825 |  12,000,265 |  948 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/keccak-914fbe99c8fbf8fe79ceba9c4b6e84b76930bdc5.md) | 18,718 |  18,655,329 |  3,348 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/sha2_bench-914fbe99c8fbf8fe79ceba9c4b6e84b76930bdc5.md) | 8,922 |  14,793,960 |  1,385 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/regex-914fbe99c8fbf8fe79ceba9c4b6e84b76930bdc5.md) | 1,417 |  4,137,067 |  378 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/ecrecover-914fbe99c8fbf8fe79ceba9c4b6e84b76930bdc5.md) | 645 |  123,583 |  268 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/pairing-914fbe99c8fbf8fe79ceba9c4b6e84b76930bdc5.md) | 910 |  1,745,757 |  286 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/kitchen_sink-914fbe99c8fbf8fe79ceba9c4b6e84b76930bdc5.md) | 2,110 |  2,579,903 |  442 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/914fbe99c8fbf8fe79ceba9c4b6e84b76930bdc5

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24597928037)
