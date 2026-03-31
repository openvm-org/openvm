| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-eefa20529775963a4136e089b8daf0855b16f93a.md) | 3,822 |  12,000,265 |  936 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-eefa20529775963a4136e089b8daf0855b16f93a.md) | 18,660 |  18,655,329 |  3,348 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-eefa20529775963a4136e089b8daf0855b16f93a.md) | 1,427 |  4,137,067 |  370 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-eefa20529775963a4136e089b8daf0855b16f93a.md) | 646 |  123,583 |  275 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-eefa20529775963a4136e089b8daf0855b16f93a.md) | 906 |  1,745,757 |  282 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-eefa20529775963a4136e089b8daf0855b16f93a.md) | 2,285 |  2,579,903 |  442 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/eefa20529775963a4136e089b8daf0855b16f93a

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23820663858)
