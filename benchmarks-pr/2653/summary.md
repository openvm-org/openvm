| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2653/fibonacci-be8419499f3acfaf3af489f0f1b7c76085a4a98c.md) | 3,809 |  12,000,265 |  937 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2653/keccak-be8419499f3acfaf3af489f0f1b7c76085a4a98c.md) | 15,573 |  1,235,218 |  2,210 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2653/regex-be8419499f3acfaf3af489f0f1b7c76085a4a98c.md) | 1,415 |  4,136,694 |  371 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2653/ecrecover-be8419499f3acfaf3af489f0f1b7c76085a4a98c.md) | 632 |  122,348 |  265 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2653/pairing-be8419499f3acfaf3af489f0f1b7c76085a4a98c.md) | 925 |  1,745,757 |  284 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2653/kitchen_sink-be8419499f3acfaf3af489f0f1b7c76085a4a98c.md) | 2,378 |  154,763 |  415 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/be8419499f3acfaf3af489f0f1b7c76085a4a98c

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23910736968)
