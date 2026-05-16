| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2788/fibonacci-430345f58eed6f0b1f5e061710521c7ab8bc2ae6.md) | 3,767 |  12,000,265 |  915 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2788/keccak-430345f58eed6f0b1f5e061710521c7ab8bc2ae6.md) | 18,531 |  18,655,329 |  3,264 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2788/sha2_bench-430345f58eed6f0b1f5e061710521c7ab8bc2ae6.md) | 10,196 |  14,793,960 |  1,463 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2788/regex-430345f58eed6f0b1f5e061710521c7ab8bc2ae6.md) | 1,397 |  4,137,067 |  353 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2788/ecrecover-430345f58eed6f0b1f5e061710521c7ab8bc2ae6.md) | 601 |  123,583 |  252 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2788/pairing-430345f58eed6f0b1f5e061710521c7ab8bc2ae6.md) | 886 |  1,745,757 |  266 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2788/kitchen_sink-430345f58eed6f0b1f5e061710521c7ab8bc2ae6.md) | 1,881 |  2,579,903 |  408 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/430345f58eed6f0b1f5e061710521c7ab8bc2ae6

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25956604096)
