| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2720/fibonacci-1fd6a4ae3d6d7b9834a45a4f88c85b510d31d1cf.md) | 3,823 |  12,000,265 |  943 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2720/keccak-1fd6a4ae3d6d7b9834a45a4f88c85b510d31d1cf.md) | 18,532 |  18,655,329 |  3,323 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2720/sha2_bench-1fd6a4ae3d6d7b9834a45a4f88c85b510d31d1cf.md) | 9,059 |  14,793,960 |  1,407 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2720/regex-1fd6a4ae3d6d7b9834a45a4f88c85b510d31d1cf.md) | 1,412 |  4,137,067 |  375 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2720/ecrecover-1fd6a4ae3d6d7b9834a45a4f88c85b510d31d1cf.md) | 651 |  123,583 |  274 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2720/pairing-1fd6a4ae3d6d7b9834a45a4f88c85b510d31d1cf.md) | 921 |  1,745,757 |  289 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2720/kitchen_sink-1fd6a4ae3d6d7b9834a45a4f88c85b510d31d1cf.md) | 2,099 |  2,579,903 |  438 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/1fd6a4ae3d6d7b9834a45a4f88c85b510d31d1cf

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24682364703)
