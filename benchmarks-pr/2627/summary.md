| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2627/fibonacci-64dda652bd42f19c208fa0a62f1b55446114859c.md) | 3,761 |  12,000,265 |  925 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2627/keccak-64dda652bd42f19c208fa0a62f1b55446114859c.md) | 15,630 |  1,235,218 |  2,196 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2627/regex-64dda652bd42f19c208fa0a62f1b55446114859c.md) | 1,419 |  4,136,694 |  376 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2627/ecrecover-64dda652bd42f19c208fa0a62f1b55446114859c.md) | 636 |  122,348 |  264 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2627/pairing-64dda652bd42f19c208fa0a62f1b55446114859c.md) | 911 |  1,745,757 |  279 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2627/kitchen_sink-64dda652bd42f19c208fa0a62f1b55446114859c.md) | 2,369 |  154,763 |  411 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/64dda652bd42f19c208fa0a62f1b55446114859c

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23662427294)
