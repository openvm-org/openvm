| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2861/fibonacci-efb532a654b2c2e67b9df496ce4147a362e36a5a.md) | 3,900 |  12,000,265 |  1,130 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2861/keccak-efb532a654b2c2e67b9df496ce4147a362e36a5a.md) | 21,751 |  18,655,329 |  4,622 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2861/sha2_bench-efb532a654b2c2e67b9df496ce4147a362e36a5a.md) | 9,646 |  14,793,960 |  1,853 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2861/regex-efb532a654b2c2e67b9df496ce4147a362e36a5a.md) | 1,513 |  4,137,067 |  429 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2861/ecrecover-efb532a654b2c2e67b9df496ce4147a362e36a5a.md) | 609 |  123,583 |  286 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2861/pairing-efb532a654b2c2e67b9df496ce4147a362e36a5a.md) | 936 |  1,745,757 |  309 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2861/kitchen_sink-efb532a654b2c2e67b9df496ce4147a362e36a5a.md) | 4,093 |  2,579,903 |  871 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/efb532a654b2c2e67b9df496ce4147a362e36a5a

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27198755267)
