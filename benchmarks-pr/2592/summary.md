| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-534de3af7e322f2fdf682cbddb32a76fe92ff8cf.md) | 3,843 |  12,000,265 |  956 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-534de3af7e322f2fdf682cbddb32a76fe92ff8cf.md) | 18,473 |  18,655,329 |  3,308 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-534de3af7e322f2fdf682cbddb32a76fe92ff8cf.md) | 1,422 |  4,137,067 |  377 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-534de3af7e322f2fdf682cbddb32a76fe92ff8cf.md) | 648 |  123,583 |  264 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-534de3af7e322f2fdf682cbddb32a76fe92ff8cf.md) | 914 |  1,745,757 |  288 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-534de3af7e322f2fdf682cbddb32a76fe92ff8cf.md) | 2,157 |  2,579,903 |  441 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/534de3af7e322f2fdf682cbddb32a76fe92ff8cf

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24159296175)
