| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2710/fibonacci-ce3f5e8b6dbeeeedfe50f789f83f32a2d706bef7.md) | 3,806 |  12,000,265 |  948 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2710/keccak-ce3f5e8b6dbeeeedfe50f789f83f32a2d706bef7.md) | 18,965 |  18,655,329 |  3,410 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2710/sha2_bench-ce3f5e8b6dbeeeedfe50f789f83f32a2d706bef7.md) | 8,961 |  14,793,960 |  1,401 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2710/regex-ce3f5e8b6dbeeeedfe50f789f83f32a2d706bef7.md) | 1,409 |  4,137,067 |  376 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2710/ecrecover-ce3f5e8b6dbeeeedfe50f789f83f32a2d706bef7.md) | 646 |  123,583 |  276 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2710/pairing-ce3f5e8b6dbeeeedfe50f789f83f32a2d706bef7.md) | 913 |  1,745,757 |  284 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2710/kitchen_sink-ce3f5e8b6dbeeeedfe50f789f83f32a2d706bef7.md) | 2,122 |  2,579,903 |  441 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/ce3f5e8b6dbeeeedfe50f789f83f32a2d706bef7

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24476718611)
