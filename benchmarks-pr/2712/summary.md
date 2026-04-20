| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/fibonacci-70b0bf989d401f86d7408cf2252c7827ba2c966d.md) | 3,876 |  12,000,265 |  958 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/keccak-70b0bf989d401f86d7408cf2252c7827ba2c966d.md) | 18,643 |  18,655,329 |  3,318 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/sha2_bench-70b0bf989d401f86d7408cf2252c7827ba2c966d.md) | 8,998 |  14,793,960 |  1,402 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/regex-70b0bf989d401f86d7408cf2252c7827ba2c966d.md) | 1,407 |  4,137,067 |  372 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/ecrecover-70b0bf989d401f86d7408cf2252c7827ba2c966d.md) | 645 |  123,583 |  273 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/pairing-70b0bf989d401f86d7408cf2252c7827ba2c966d.md) | 903 |  1,745,757 |  284 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/kitchen_sink-70b0bf989d401f86d7408cf2252c7827ba2c966d.md) | 2,093 |  2,579,903 |  437 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/70b0bf989d401f86d7408cf2252c7827ba2c966d

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24681580816)
