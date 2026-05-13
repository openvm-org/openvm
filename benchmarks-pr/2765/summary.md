| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/fibonacci-397f4697dbef5747dd856d909106506ad94ef571.md) | 1,906 |  4,000,051 |  535 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/keccak-397f4697dbef5747dd856d909106506ad94ef571.md) | 13,587 |  14,365,133 |  2,227 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/sha2_bench-397f4697dbef5747dd856d909106506ad94ef571.md) | 9,415 |  11,167,961 |  1,400 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/regex-397f4697dbef5747dd856d909106506ad94ef571.md) | 1,612 |  4,090,656 |  380 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/ecrecover-397f4697dbef5747dd856d909106506ad94ef571.md) | 641 |  112,210 |  282 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/pairing-397f4697dbef5747dd856d909106506ad94ef571.md) | 754 |  592,827 |  281 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/kitchen_sink-397f4697dbef5747dd856d909106506ad94ef571.md) | 2,028 |  1,979,971 |  431 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/397f4697dbef5747dd856d909106506ad94ef571

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25826742532)
