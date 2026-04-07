| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2669/fibonacci-2fd57b0af15430d13f94b487e8fa18e6545c651a.md) | 3,856 |  12,000,265 |  960 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2669/keccak-2fd57b0af15430d13f94b487e8fa18e6545c651a.md) | 18,181 |  18,655,329 |  3,261 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2669/regex-2fd57b0af15430d13f94b487e8fa18e6545c651a.md) | 1,429 |  4,137,067 |  375 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2669/ecrecover-2fd57b0af15430d13f94b487e8fa18e6545c651a.md) | 667 |  123,583 |  274 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2669/pairing-2fd57b0af15430d13f94b487e8fa18e6545c651a.md) | 905 |  1,745,757 |  282 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2669/kitchen_sink-2fd57b0af15430d13f94b487e8fa18e6545c651a.md) | 2,157 |  2,579,903 |  437 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/2fd57b0af15430d13f94b487e8fa18e6545c651a

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24104932173)
