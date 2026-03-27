| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2627/fibonacci-0ef78a0591fd446dc2c689f2f902d81a50f665ec.md) | 3,753 |  12,000,265 |  922 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2627/keccak-0ef78a0591fd446dc2c689f2f902d81a50f665ec.md) | 15,661 |  1,235,218 |  2,182 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2627/regex-0ef78a0591fd446dc2c689f2f902d81a50f665ec.md) | 1,408 |  4,136,694 |  367 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2627/ecrecover-0ef78a0591fd446dc2c689f2f902d81a50f665ec.md) | 637 |  122,348 |  266 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2627/pairing-0ef78a0591fd446dc2c689f2f902d81a50f665ec.md) | 926 |  1,745,757 |  284 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2627/kitchen_sink-0ef78a0591fd446dc2c689f2f902d81a50f665ec.md) | 2,370 |  154,763 |  415 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/0ef78a0591fd446dc2c689f2f902d81a50f665ec

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23668217469)
