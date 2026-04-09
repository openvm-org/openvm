| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2683/fibonacci-35a7b82817f89761f3a3cb5e383445aafe18a2e2.md) | 3,796 |  12,000,265 |  943 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2683/keccak-35a7b82817f89761f3a3cb5e383445aafe18a2e2.md) | 18,416 |  18,655,329 |  3,309 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2683/regex-35a7b82817f89761f3a3cb5e383445aafe18a2e2.md) | 1,438 |  4,137,067 |  382 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2683/ecrecover-35a7b82817f89761f3a3cb5e383445aafe18a2e2.md) | 652 |  123,583 |  273 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2683/pairing-35a7b82817f89761f3a3cb5e383445aafe18a2e2.md) | 908 |  1,745,757 |  283 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2683/kitchen_sink-35a7b82817f89761f3a3cb5e383445aafe18a2e2.md) | 2,151 |  2,579,903 |  437 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/35a7b82817f89761f3a3cb5e383445aafe18a2e2

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24202380111)
