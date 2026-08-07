| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3099/fibonacci-5b4122ddde9adaea84882cd37d433dd2a82743dc.md) | 1,577 |  12,000,265 |  360 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3099/keccak-5b4122ddde9adaea84882cd37d433dd2a82743dc.md) | 9,384 |  18,655,329 |  1,548 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3099/sha2_bench-5b4122ddde9adaea84882cd37d433dd2a82743dc.md) | 4,953 |  14,793,960 |  571 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3099/regex-5b4122ddde9adaea84882cd37d433dd2a82743dc.md) | 665 |  4,137,067 |  220 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3099/ecrecover-5b4122ddde9adaea84882cd37d433dd2a82743dc.md) | 435 |  123,583 |  187 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3099/pairing-5b4122ddde9adaea84882cd37d433dd2a82743dc.md) | 560 |  1,745,757 |  191 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3099/kitchen_sink-5b4122ddde9adaea84882cd37d433dd2a82743dc.md) | 2,204 |  2,579,903 |  479 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/5b4122ddde9adaea84882cd37d433dd2a82743dc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31141801188)
