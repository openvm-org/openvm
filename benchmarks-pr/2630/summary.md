| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2630/fibonacci-cc37d1c7a9eb6a0bbe92f9e489dc5f494af4caf9.md) | 3,865 |  12,000,265 |  953 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2630/keccak-cc37d1c7a9eb6a0bbe92f9e489dc5f494af4caf9.md) | 18,456 |  18,655,329 |  3,336 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2630/regex-cc37d1c7a9eb6a0bbe92f9e489dc5f494af4caf9.md) | 1,416 |  4,137,067 |  369 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2630/ecrecover-cc37d1c7a9eb6a0bbe92f9e489dc5f494af4caf9.md) | 647 |  123,583 |  268 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2630/pairing-cc37d1c7a9eb6a0bbe92f9e489dc5f494af4caf9.md) | 899 |  1,745,757 |  284 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2630/kitchen_sink-cc37d1c7a9eb6a0bbe92f9e489dc5f494af4caf9.md) | 2,293 |  2,579,903 |  443 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/cc37d1c7a9eb6a0bbe92f9e489dc5f494af4caf9

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23804685426)
