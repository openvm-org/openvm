| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2646/fibonacci-d96449eaaa0a469a246086b52f6c2f00bc52440c.md) | 3,829 |  12,000,265 |  935 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2646/keccak-d96449eaaa0a469a246086b52f6c2f00bc52440c.md) | 18,549 |  18,655,329 |  3,303 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2646/regex-d96449eaaa0a469a246086b52f6c2f00bc52440c.md) | 1,419 |  4,137,067 |  370 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2646/ecrecover-d96449eaaa0a469a246086b52f6c2f00bc52440c.md) | 735 |  317,792 |  353 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2646/pairing-d96449eaaa0a469a246086b52f6c2f00bc52440c.md) | 917 |  1,745,757 |  317 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2646/kitchen_sink-d96449eaaa0a469a246086b52f6c2f00bc52440c.md) | 2,502 |  2,580,026 |  544 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/d96449eaaa0a469a246086b52f6c2f00bc52440c

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23900987270)
