| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2764/fibonacci-5735596bc65da1602729bcb7a76e5a6fc6176b3f.md) | 3,784 |  12,000,265 |  956 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2764/keccak-5735596bc65da1602729bcb7a76e5a6fc6176b3f.md) | 19,137 |  18,655,329 |  3,382 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2764/sha2_bench-5735596bc65da1602729bcb7a76e5a6fc6176b3f.md) | 9,056 |  14,793,960 |  1,398 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2764/regex-5735596bc65da1602729bcb7a76e5a6fc6176b3f.md) | 1,424 |  4,137,067 |  375 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2764/ecrecover-5735596bc65da1602729bcb7a76e5a6fc6176b3f.md) | 641 |  123,583 |  274 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2764/pairing-5735596bc65da1602729bcb7a76e5a6fc6176b3f.md) | 905 |  1,745,757 |  283 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2764/kitchen_sink-5735596bc65da1602729bcb7a76e5a6fc6176b3f.md) | 2,039 |  2,579,903 |  433 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/5735596bc65da1602729bcb7a76e5a6fc6176b3f

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25179840395)
