| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2625/fibonacci-80238388a4f83d2b1a19b1aac37bddb8e2e3d72a.md) | 3,812 |  12,000,265 |  944 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2625/keccak-80238388a4f83d2b1a19b1aac37bddb8e2e3d72a.md) | 15,800 |  1,235,218 |  2,190 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2625/regex-80238388a4f83d2b1a19b1aac37bddb8e2e3d72a.md) | 1,415 |  4,136,694 |  365 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2625/ecrecover-80238388a4f83d2b1a19b1aac37bddb8e2e3d72a.md) | 640 |  122,348 |  269 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2625/pairing-80238388a4f83d2b1a19b1aac37bddb8e2e3d72a.md) | 919 |  1,745,757 |  282 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2625/kitchen_sink-80238388a4f83d2b1a19b1aac37bddb8e2e3d72a.md) | 2,376 |  154,763 |  405 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/80238388a4f83d2b1a19b1aac37bddb8e2e3d72a

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23816783518)
