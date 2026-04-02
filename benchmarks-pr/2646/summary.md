| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2646/fibonacci-58582a0ba125d6de23064bf8e644ee3b37798feb.md) | 3,826 |  12,000,265 |  940 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2646/keccak-58582a0ba125d6de23064bf8e644ee3b37798feb.md) | 19,024 |  18,655,329 |  3,399 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2646/regex-58582a0ba125d6de23064bf8e644ee3b37798feb.md) | 1,417 |  4,137,067 |  375 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2646/ecrecover-58582a0ba125d6de23064bf8e644ee3b37798feb.md) | 738 |  317,792 |  345 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2646/pairing-58582a0ba125d6de23064bf8e644ee3b37798feb.md) | 907 |  1,745,757 |  314 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2646/kitchen_sink-58582a0ba125d6de23064bf8e644ee3b37798feb.md) | 2,492 |  2,580,026 |  542 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/58582a0ba125d6de23064bf8e644ee3b37798feb

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23904092634)
