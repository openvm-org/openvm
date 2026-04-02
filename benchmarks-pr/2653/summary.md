| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2653/fibonacci-9fae58d796a4663e68c675efcd207f71258bed21.md) | 3,842 |  12,000,265 |  949 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2653/keccak-9fae58d796a4663e68c675efcd207f71258bed21.md) | 15,784 |  1,235,218 |  2,219 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2653/regex-9fae58d796a4663e68c675efcd207f71258bed21.md) | 1,418 |  4,136,694 |  372 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2653/ecrecover-9fae58d796a4663e68c675efcd207f71258bed21.md) | 634 |  122,348 |  268 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2653/pairing-9fae58d796a4663e68c675efcd207f71258bed21.md) | 918 |  1,745,757 |  278 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2653/kitchen_sink-9fae58d796a4663e68c675efcd207f71258bed21.md) | 2,365 |  154,763 |  410 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/9fae58d796a4663e68c675efcd207f71258bed21

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23922415143)
