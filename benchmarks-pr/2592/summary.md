| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-9bad1c55942d010fe7691c3d8baa26a5352a88fa.md) | 3,815 |  12,000,265 |  948 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-9bad1c55942d010fe7691c3d8baa26a5352a88fa.md) | 18,645 |  18,655,329 |  3,325 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-9bad1c55942d010fe7691c3d8baa26a5352a88fa.md) | 1,413 |  4,137,067 |  374 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-9bad1c55942d010fe7691c3d8baa26a5352a88fa.md) | 644 |  123,583 |  272 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-9bad1c55942d010fe7691c3d8baa26a5352a88fa.md) | 905 |  1,745,757 |  282 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-9bad1c55942d010fe7691c3d8baa26a5352a88fa.md) | 2,153 |  2,579,903 |  437 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/9bad1c55942d010fe7691c3d8baa26a5352a88fa

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24182248701)
