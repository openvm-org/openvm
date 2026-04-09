| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2681/fibonacci-647366cf5e5c5145c40ea1fccd30a6581eacb3ad.md) | 3,813 |  12,000,265 |  947 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2681/keccak-647366cf5e5c5145c40ea1fccd30a6581eacb3ad.md) | 18,496 |  18,655,329 |  3,319 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2681/regex-647366cf5e5c5145c40ea1fccd30a6581eacb3ad.md) | 1,419 |  4,137,067 |  373 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2681/ecrecover-647366cf5e5c5145c40ea1fccd30a6581eacb3ad.md) | 649 |  123,583 |  272 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2681/pairing-647366cf5e5c5145c40ea1fccd30a6581eacb3ad.md) | 901 |  1,745,757 |  282 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2681/kitchen_sink-647366cf5e5c5145c40ea1fccd30a6581eacb3ad.md) | 2,160 |  2,579,903 |  434 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/647366cf5e5c5145c40ea1fccd30a6581eacb3ad

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24182032906)
