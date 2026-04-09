| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2674/fibonacci-bef2c5ee96c0c007a5b1684c9a0d1e5178215281.md) | 3,823 |  12,000,265 |  952 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2674/keccak-bef2c5ee96c0c007a5b1684c9a0d1e5178215281.md) | 18,894 |  18,655,329 |  3,374 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2674/regex-bef2c5ee96c0c007a5b1684c9a0d1e5178215281.md) | 1,412 |  4,137,067 |  373 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2674/ecrecover-bef2c5ee96c0c007a5b1684c9a0d1e5178215281.md) | 649 |  123,583 |  272 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2674/pairing-bef2c5ee96c0c007a5b1684c9a0d1e5178215281.md) | 899 |  1,745,757 |  278 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2674/kitchen_sink-bef2c5ee96c0c007a5b1684c9a0d1e5178215281.md) | 2,154 |  2,579,903 |  437 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/bef2c5ee96c0c007a5b1684c9a0d1e5178215281

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24203663640)
