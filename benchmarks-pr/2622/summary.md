| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2622/fibonacci-291d4eb7b2b46b7ee69d4b8bf68656e8ba3c2132.md) | 3,827 |  12,000,265 |  941 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2622/keccak-291d4eb7b2b46b7ee69d4b8bf68656e8ba3c2132.md) | 15,721 |  1,235,218 |  2,188 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2622/regex-291d4eb7b2b46b7ee69d4b8bf68656e8ba3c2132.md) | 1,416 |  4,136,694 |  372 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2622/ecrecover-291d4eb7b2b46b7ee69d4b8bf68656e8ba3c2132.md) | 633 |  122,348 |  267 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2622/pairing-291d4eb7b2b46b7ee69d4b8bf68656e8ba3c2132.md) | 913 |  1,745,757 |  284 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2622/kitchen_sink-291d4eb7b2b46b7ee69d4b8bf68656e8ba3c2132.md) | 2,368 |  154,763 |  405 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/291d4eb7b2b46b7ee69d4b8bf68656e8ba3c2132

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23594416680)
