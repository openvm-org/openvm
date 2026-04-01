| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-87f486ccd44ff9922e9b62fc3c62a581c49e8c7d.md) | 3,887 |  12,000,265 |  947 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-87f486ccd44ff9922e9b62fc3c62a581c49e8c7d.md) | 18,396 |  18,655,329 |  3,278 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-87f486ccd44ff9922e9b62fc3c62a581c49e8c7d.md) | 1,433 |  4,137,067 |  376 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-87f486ccd44ff9922e9b62fc3c62a581c49e8c7d.md) | 642 |  123,583 |  265 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-87f486ccd44ff9922e9b62fc3c62a581c49e8c7d.md) | 897 |  1,745,757 |  282 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-87f486ccd44ff9922e9b62fc3c62a581c49e8c7d.md) | 2,279 |  2,579,903 |  442 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/87f486ccd44ff9922e9b62fc3c62a581c49e8c7d

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23870238749)
