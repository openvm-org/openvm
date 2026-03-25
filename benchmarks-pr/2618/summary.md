| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2618/fibonacci-e395b986569fa29a5511c0bfa3ba5c6cd71de734.md) | 4,135 |  12,000,265 |  1,359 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2618/keccak-e395b986569fa29a5511c0bfa3ba5c6cd71de734.md) | 19,318 |  1,235,218 |  3,382 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2618/regex-e395b986569fa29a5511c0bfa3ba5c6cd71de734.md) | 1,618 |  4,136,694 |  533 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2618/ecrecover-e395b986569fa29a5511c0bfa3ba5c6cd71de734.md) | 649 |  122,348 |  340 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2618/pairing-e395b986569fa29a5511c0bfa3ba5c6cd71de734.md) | 1,059 |  1,745,757 |  343 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2618/kitchen_sink-e395b986569fa29a5511c0bfa3ba5c6cd71de734.md) | 3,294 |  154,763 |  723 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/e395b986569fa29a5511c0bfa3ba5c6cd71de734

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23558003722)
