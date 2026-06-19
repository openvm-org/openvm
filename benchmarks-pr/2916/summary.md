| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2916/fibonacci-612457a2fd5e88772de18c38b6570a29a9143d12.md) | 1,052 |  4,000,051 |  395 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2916/keccak-612457a2fd5e88772de18c38b6570a29a9143d12.md) | 16,218 |  14,365,133 |  3,008 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2916/sha2_bench-612457a2fd5e88772de18c38b6570a29a9143d12.md) | 8,187 |  11,167,961 |  995 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2916/regex-612457a2fd5e88772de18c38b6570a29a9143d12.md) | 1,204 |  4,090,656 |  359 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2916/ecrecover-612457a2fd5e88772de18c38b6570a29a9143d12.md) | 433 |  112,210 |  274 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2916/pairing-612457a2fd5e88772de18c38b6570a29a9143d12.md) | 605 |  592,827 |  298 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2916/kitchen_sink-612457a2fd5e88772de18c38b6570a29a9143d12.md) | 3,894 |  1,979,971 |  865 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/612457a2fd5e88772de18c38b6570a29a9143d12

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27850519747)
