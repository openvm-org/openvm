| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2679/fibonacci-e04f5a0ec7b1f1d7eff702d56bc14315557730f4.md) | 3,813 |  12,000,265 |  947 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2679/keccak-e04f5a0ec7b1f1d7eff702d56bc14315557730f4.md) | 18,388 |  18,655,329 |  3,294 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2679/regex-e04f5a0ec7b1f1d7eff702d56bc14315557730f4.md) | 1,412 |  4,137,067 |  375 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2679/ecrecover-e04f5a0ec7b1f1d7eff702d56bc14315557730f4.md) | 647 |  123,583 |  270 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2679/pairing-e04f5a0ec7b1f1d7eff702d56bc14315557730f4.md) | 915 |  1,745,757 |  288 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2679/kitchen_sink-e04f5a0ec7b1f1d7eff702d56bc14315557730f4.md) | 2,163 |  2,579,903 |  440 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/e04f5a0ec7b1f1d7eff702d56bc14315557730f4

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24178679874)
