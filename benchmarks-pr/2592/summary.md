| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-37671959b4e9cb596b4cd491ba1aa2a34621edd8.md) | 3,836 |  12,000,265 |  938 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-37671959b4e9cb596b4cd491ba1aa2a34621edd8.md) | 18,583 |  18,655,329 |  3,318 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-37671959b4e9cb596b4cd491ba1aa2a34621edd8.md) | 1,414 |  4,137,067 |  370 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-37671959b4e9cb596b4cd491ba1aa2a34621edd8.md) | 658 |  123,583 |  272 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-37671959b4e9cb596b4cd491ba1aa2a34621edd8.md) | 908 |  1,745,757 |  284 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-37671959b4e9cb596b4cd491ba1aa2a34621edd8.md) | 2,283 |  2,579,903 |  436 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/37671959b4e9cb596b4cd491ba1aa2a34621edd8

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23959075217)
