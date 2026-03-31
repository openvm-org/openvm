| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2636/fibonacci-6be5b9f53cd9641779b9e8ae2d98ac303c876cdc.md) | 3,846 |  12,000,265 |  947 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2636/keccak-6be5b9f53cd9641779b9e8ae2d98ac303c876cdc.md) | 15,702 |  1,235,218 |  2,205 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2636/regex-6be5b9f53cd9641779b9e8ae2d98ac303c876cdc.md) | 1,434 |  4,136,694 |  373 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2636/ecrecover-6be5b9f53cd9641779b9e8ae2d98ac303c876cdc.md) | 635 |  122,348 |  270 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2636/pairing-6be5b9f53cd9641779b9e8ae2d98ac303c876cdc.md) | 925 |  1,745,757 |  281 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2636/kitchen_sink-6be5b9f53cd9641779b9e8ae2d98ac303c876cdc.md) | 2,373 |  154,763 |  417 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/6be5b9f53cd9641779b9e8ae2d98ac303c876cdc

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23818128769)
