| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2668/fibonacci-0473bff96d67258421ae6265284927e5b87902d6.md) | 3,851 |  12,000,265 |  958 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2668/keccak-0473bff96d67258421ae6265284927e5b87902d6.md) | 15,735 |  1,235,218 |  2,214 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2668/regex-0473bff96d67258421ae6265284927e5b87902d6.md) | 1,420 |  4,136,694 |  371 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2668/ecrecover-0473bff96d67258421ae6265284927e5b87902d6.md) | 639 |  122,348 |  268 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2668/pairing-0473bff96d67258421ae6265284927e5b87902d6.md) | 917 |  1,745,757 |  286 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2668/kitchen_sink-0473bff96d67258421ae6265284927e5b87902d6.md) | 2,372 |  154,763 |  416 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/0473bff96d67258421ae6265284927e5b87902d6

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24104621515)
