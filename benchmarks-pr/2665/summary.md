| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2665/fibonacci-ec7b84d10e17d924ee3a56bfacda31e44964e145.md) | 3,837 |  12,000,265 |  952 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2665/keccak-ec7b84d10e17d924ee3a56bfacda31e44964e145.md) | 18,409 |  18,655,329 |  3,309 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2665/regex-ec7b84d10e17d924ee3a56bfacda31e44964e145.md) | 1,417 |  4,137,067 |  378 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2665/ecrecover-ec7b84d10e17d924ee3a56bfacda31e44964e145.md) | 667 |  123,583 |  273 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2665/pairing-ec7b84d10e17d924ee3a56bfacda31e44964e145.md) | 907 |  1,745,757 |  283 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2665/kitchen_sink-ec7b84d10e17d924ee3a56bfacda31e44964e145.md) | 2,299 |  2,579,903 |  441 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/ec7b84d10e17d924ee3a56bfacda31e44964e145

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24055831549)
