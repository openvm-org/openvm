| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2763/fibonacci-15bae4210958f13f759f71f93485ff4dfe7a16cf.md) | 3,770 |  12,000,265 |  936 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2763/keccak-15bae4210958f13f759f71f93485ff4dfe7a16cf.md) | 18,692 |  18,655,329 |  3,335 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2763/sha2_bench-15bae4210958f13f759f71f93485ff4dfe7a16cf.md) | 8,954 |  14,793,960 |  1,402 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2763/regex-15bae4210958f13f759f71f93485ff4dfe7a16cf.md) | 1,397 |  4,137,067 |  375 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2763/ecrecover-15bae4210958f13f759f71f93485ff4dfe7a16cf.md) | 636 |  123,583 |  273 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2763/pairing-15bae4210958f13f759f71f93485ff4dfe7a16cf.md) | 893 |  1,745,757 |  281 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2763/kitchen_sink-15bae4210958f13f759f71f93485ff4dfe7a16cf.md) | 2,088 |  2,579,903 |  436 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/15bae4210958f13f759f71f93485ff4dfe7a16cf

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25174706723)
