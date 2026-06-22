| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2919/fibonacci-008ad62ea37613fd7d1be0037d22e1cae76d2490.md) | 3,069 |  12,000,265 |  677 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2919/keccak-008ad62ea37613fd7d1be0037d22e1cae76d2490.md) | 16,345 |  18,655,329 |  3,037 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2919/sha2_bench-008ad62ea37613fd7d1be0037d22e1cae76d2490.md) | 9,125 |  14,793,960 |  1,120 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2919/regex-008ad62ea37613fd7d1be0037d22e1cae76d2490.md) | 1,164 |  4,137,067 |  350 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2919/ecrecover-008ad62ea37613fd7d1be0037d22e1cae76d2490.md) | 604 |  123,583 |  282 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2919/pairing-008ad62ea37613fd7d1be0037d22e1cae76d2490.md) | 945 |  1,745,757 |  309 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2919/kitchen_sink-008ad62ea37613fd7d1be0037d22e1cae76d2490.md) | 4,109 |  2,579,903 |  878 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/008ad62ea37613fd7d1be0037d22e1cae76d2490

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27966224865)
