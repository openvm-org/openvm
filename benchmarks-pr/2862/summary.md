| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2862/fibonacci-4c33cf70b986023280907f6f57b1b6e4b67286ad.md) | 3,746 |  12,000,265 |  920 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2862/keccak-4c33cf70b986023280907f6f57b1b6e4b67286ad.md) | 18,026 |  18,655,329 |  3,277 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2862/sha2_bench-4c33cf70b986023280907f6f57b1b6e4b67286ad.md) | 9,949 |  14,793,960 |  1,448 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2862/regex-4c33cf70b986023280907f6f57b1b6e4b67286ad.md) | 1,394 |  4,137,067 |  358 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2862/ecrecover-4c33cf70b986023280907f6f57b1b6e4b67286ad.md) | 596 |  123,583 |  253 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2862/pairing-4c33cf70b986023280907f6f57b1b6e4b67286ad.md) | 880 |  1,745,757 |  260 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2862/kitchen_sink-4c33cf70b986023280907f6f57b1b6e4b67286ad.md) | 3,869 |  2,579,903 |  967 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/4c33cf70b986023280907f6f57b1b6e4b67286ad

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27220507840)
