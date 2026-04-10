| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2678/fibonacci-728df70b6cd0c17d88097fb9824b69c229882319.md) | 3,831 |  12,000,265 |  946 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2678/keccak-728df70b6cd0c17d88097fb9824b69c229882319.md) | 18,438 |  18,655,329 |  3,302 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2678/regex-728df70b6cd0c17d88097fb9824b69c229882319.md) | 1,425 |  4,137,067 |  378 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2678/ecrecover-728df70b6cd0c17d88097fb9824b69c229882319.md) | 643 |  123,583 |  270 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2678/pairing-728df70b6cd0c17d88097fb9824b69c229882319.md) | 907 |  1,745,757 |  284 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2678/kitchen_sink-728df70b6cd0c17d88097fb9824b69c229882319.md) | 2,166 |  2,579,903 |  432 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/728df70b6cd0c17d88097fb9824b69c229882319

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24258642967)
