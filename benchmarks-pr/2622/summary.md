| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2622/fibonacci-22c9ef2c401b67300d7b5f05c8d9b94ef2ffa28c.md) | 3,820 |  12,000,265 |  935 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2622/keccak-22c9ef2c401b67300d7b5f05c8d9b94ef2ffa28c.md) | 15,735 |  1,235,218 |  2,178 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2622/regex-22c9ef2c401b67300d7b5f05c8d9b94ef2ffa28c.md) | 1,434 |  4,136,694 |  384 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2622/ecrecover-22c9ef2c401b67300d7b5f05c8d9b94ef2ffa28c.md) | 637 |  122,348 |  269 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2622/pairing-22c9ef2c401b67300d7b5f05c8d9b94ef2ffa28c.md) | 920 |  1,745,757 |  288 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2622/kitchen_sink-22c9ef2c401b67300d7b5f05c8d9b94ef2ffa28c.md) | 2,370 |  154,763 |  408 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/22c9ef2c401b67300d7b5f05c8d9b94ef2ffa28c

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23611944135)
