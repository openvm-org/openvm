| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/fibonacci-b96c15a3d84d6c78a23ea82e585b039ac480f076.md) | 3,070 |  12,000,265 |  671 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/keccak-b96c15a3d84d6c78a23ea82e585b039ac480f076.md) | 16,276 |  18,655,329 |  3,021 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/sha2_bench-b96c15a3d84d6c78a23ea82e585b039ac480f076.md) | 9,207 |  14,793,960 |  1,123 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/regex-b96c15a3d84d6c78a23ea82e585b039ac480f076.md) | 1,178 |  4,137,067 |  355 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/ecrecover-b96c15a3d84d6c78a23ea82e585b039ac480f076.md) | 600 |  123,583 |  280 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/pairing-b96c15a3d84d6c78a23ea82e585b039ac480f076.md) | 945 |  1,745,757 |  299 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2889/kitchen_sink-b96c15a3d84d6c78a23ea82e585b039ac480f076.md) | 4,157 |  2,579,903 |  887 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/b96c15a3d84d6c78a23ea82e585b039ac480f076

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27971821102)
