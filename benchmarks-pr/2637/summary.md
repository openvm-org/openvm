| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2637/fibonacci-931b0eac7a5f97eaafe0c4a928a01ff838f4bb7d.md) | 3,895 |  12,000,265 |  966 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2637/keccak-931b0eac7a5f97eaafe0c4a928a01ff838f4bb7d.md) | 15,645 |  1,235,218 |  2,192 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2637/regex-931b0eac7a5f97eaafe0c4a928a01ff838f4bb7d.md) | 1,411 |  4,136,694 |  371 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2637/ecrecover-931b0eac7a5f97eaafe0c4a928a01ff838f4bb7d.md) | 635 |  122,348 |  270 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2637/pairing-931b0eac7a5f97eaafe0c4a928a01ff838f4bb7d.md) | 917 |  1,745,757 |  278 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2637/kitchen_sink-931b0eac7a5f97eaafe0c4a928a01ff838f4bb7d.md) | 2,369 |  154,763 |  415 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/931b0eac7a5f97eaafe0c4a928a01ff838f4bb7d

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23815464297)
