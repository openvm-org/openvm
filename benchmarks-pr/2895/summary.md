| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2895/fibonacci-af3a6287087dee6fde826414b6c9e72be350313c.md) | 3,064 |  12,000,265 |  674 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2895/keccak-af3a6287087dee6fde826414b6c9e72be350313c.md) | 16,274 |  18,655,329 |  3,019 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2895/sha2_bench-af3a6287087dee6fde826414b6c9e72be350313c.md) | 9,028 |  14,793,960 |  1,106 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2895/regex-af3a6287087dee6fde826414b6c9e72be350313c.md) | 1,156 |  4,137,067 |  349 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2895/ecrecover-af3a6287087dee6fde826414b6c9e72be350313c.md) | 599 |  123,583 |  282 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2895/pairing-af3a6287087dee6fde826414b6c9e72be350313c.md) | 926 |  1,745,757 |  303 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2895/kitchen_sink-af3a6287087dee6fde826414b6c9e72be350313c.md) | 4,122 |  2,579,903 |  879 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/af3a6287087dee6fde826414b6c9e72be350313c

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27624409073)
