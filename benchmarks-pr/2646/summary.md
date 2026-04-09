| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2646/fibonacci-139e261b5395fda64c833b0e0ae29338291750c7.md) | 3,820 |  12,000,265 |  945 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2646/keccak-139e261b5395fda64c833b0e0ae29338291750c7.md) | 18,648 |  18,655,329 |  3,338 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2646/regex-139e261b5395fda64c833b0e0ae29338291750c7.md) | 1,423 |  4,137,067 |  378 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2646/ecrecover-139e261b5395fda64c833b0e0ae29338291750c7.md) | 735 |  317,792 |  351 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2646/pairing-139e261b5395fda64c833b0e0ae29338291750c7.md) | 920 |  1,745,757 |  318 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2646/kitchen_sink-139e261b5395fda64c833b0e0ae29338291750c7.md) | 2,355 |  2,580,026 |  776 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/139e261b5395fda64c833b0e0ae29338291750c7

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24214882040)
