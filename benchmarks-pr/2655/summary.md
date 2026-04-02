| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2655/fibonacci-d0a359e80ebebe3900eee304cdb4a3fa6082e262.md) | 3,803 |  12,000,265 |  936 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2655/keccak-d0a359e80ebebe3900eee304cdb4a3fa6082e262.md) | 15,565 |  1,235,218 |  2,178 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2655/regex-d0a359e80ebebe3900eee304cdb4a3fa6082e262.md) | 1,423 |  4,136,694 |  370 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2655/ecrecover-d0a359e80ebebe3900eee304cdb4a3fa6082e262.md) | 637 |  122,348 |  266 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2655/pairing-d0a359e80ebebe3900eee304cdb4a3fa6082e262.md) | 914 |  1,745,757 |  282 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2655/kitchen_sink-d0a359e80ebebe3900eee304cdb4a3fa6082e262.md) | 2,374 |  154,763 |  414 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/d0a359e80ebebe3900eee304cdb4a3fa6082e262

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23915439629)
