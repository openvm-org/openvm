| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-863127a8dc5eecd5babc2aa34636031353a85239.md) | 3,906 |  12,000,265 |  964 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-863127a8dc5eecd5babc2aa34636031353a85239.md) | 18,799 |  18,655,329 |  3,360 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-863127a8dc5eecd5babc2aa34636031353a85239.md) | 1,444 |  4,137,067 |  376 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-863127a8dc5eecd5babc2aa34636031353a85239.md) | 647 |  123,583 |  272 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-863127a8dc5eecd5babc2aa34636031353a85239.md) | 899 |  1,745,757 |  281 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-863127a8dc5eecd5babc2aa34636031353a85239.md) | 2,295 |  2,579,903 |  442 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/863127a8dc5eecd5babc2aa34636031353a85239

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23860571361)
