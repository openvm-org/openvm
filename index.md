| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/fibonacci-902c4a4118bfaa3e492f6abeea24cb658ee8de7d.md) | 1,661 |  12,000,265 |  373 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/keccak-902c4a4118bfaa3e492f6abeea24cb658ee8de7d.md) | 9,526 |  18,655,329 |  1,545 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/sha2_bench-902c4a4118bfaa3e492f6abeea24cb658ee8de7d.md) | 5,266 |  14,793,960 |  588 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/regex-902c4a4118bfaa3e492f6abeea24cb658ee8de7d.md) | 699 |  4,137,067 |  222 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/ecrecover-902c4a4118bfaa3e492f6abeea24cb658ee8de7d.md) | 430 |  123,583 |  192 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/pairing-902c4a4118bfaa3e492f6abeea24cb658ee8de7d.md) | 610 |  1,745,757 |  197 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/kitchen_sink-902c4a4118bfaa3e492f6abeea24cb658ee8de7d.md) | 2,317 |  2,579,903 |  494 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/902c4a4118bfaa3e492f6abeea24cb658ee8de7d

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/33100151507)
