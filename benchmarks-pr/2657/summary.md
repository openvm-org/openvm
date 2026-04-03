| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2657/fibonacci-46e2f4e886e2bda251d0aa1e5b024cc4e5dea5bf.md) | 3,811 |  12,000,265 |  941 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2657/keccak-46e2f4e886e2bda251d0aa1e5b024cc4e5dea5bf.md) | 15,842 |  1,235,218 |  2,229 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2657/regex-46e2f4e886e2bda251d0aa1e5b024cc4e5dea5bf.md) | 1,423 |  4,136,694 |  369 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2657/ecrecover-46e2f4e886e2bda251d0aa1e5b024cc4e5dea5bf.md) | 644 |  122,348 |  266 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2657/pairing-46e2f4e886e2bda251d0aa1e5b024cc4e5dea5bf.md) | 909 |  1,745,757 |  280 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2657/kitchen_sink-46e2f4e886e2bda251d0aa1e5b024cc4e5dea5bf.md) | 2,369 |  154,763 |  413 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/46e2f4e886e2bda251d0aa1e5b024cc4e5dea5bf

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23928369211)
