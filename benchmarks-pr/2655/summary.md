| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2655/fibonacci-d0be9e35cc1022f4ccbfe1fe215c11b979c33c83.md) | 3,836 |  12,000,265 |  941 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2655/keccak-d0be9e35cc1022f4ccbfe1fe215c11b979c33c83.md) | 15,523 |  1,235,218 |  2,167 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2655/regex-d0be9e35cc1022f4ccbfe1fe215c11b979c33c83.md) | 1,425 |  4,136,694 |  368 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2655/ecrecover-d0be9e35cc1022f4ccbfe1fe215c11b979c33c83.md) | 637 |  122,348 |  266 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2655/pairing-d0be9e35cc1022f4ccbfe1fe215c11b979c33c83.md) | 912 |  1,745,757 |  276 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2655/kitchen_sink-d0be9e35cc1022f4ccbfe1fe215c11b979c33c83.md) | 2,377 |  154,763 |  417 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/d0be9e35cc1022f4ccbfe1fe215c11b979c33c83

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23922230579)
