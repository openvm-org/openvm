| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/fibonacci-0f30aaa5fc80082056f6e7aa801637b3af1a0365.md) | 3,820 |  12,000,265 |  951 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/keccak-0f30aaa5fc80082056f6e7aa801637b3af1a0365.md) | 18,675 |  18,655,329 |  3,352 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/regex-0f30aaa5fc80082056f6e7aa801637b3af1a0365.md) | 1,407 |  4,137,067 |  372 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/ecrecover-0f30aaa5fc80082056f6e7aa801637b3af1a0365.md) | 648 |  123,583 |  268 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/pairing-0f30aaa5fc80082056f6e7aa801637b3af1a0365.md) | 908 |  1,745,757 |  286 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2662/kitchen_sink-0f30aaa5fc80082056f6e7aa801637b3af1a0365.md) | 2,153 |  2,579,903 |  436 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/0f30aaa5fc80082056f6e7aa801637b3af1a0365

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24254023075)
