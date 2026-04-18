| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/fibonacci-fff47adc7bb8b6f1f5820728ac9189c0e0aa9672.md) | 3,863 |  12,000,265 |  960 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/keccak-fff47adc7bb8b6f1f5820728ac9189c0e0aa9672.md) | 18,455 |  18,655,329 |  3,300 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/sha2_bench-fff47adc7bb8b6f1f5820728ac9189c0e0aa9672.md) | 8,971 |  14,793,960 |  1,395 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/regex-fff47adc7bb8b6f1f5820728ac9189c0e0aa9672.md) | 1,427 |  4,137,067 |  376 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/ecrecover-fff47adc7bb8b6f1f5820728ac9189c0e0aa9672.md) | 650 |  123,583 |  272 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/pairing-fff47adc7bb8b6f1f5820728ac9189c0e0aa9672.md) | 909 |  1,745,757 |  285 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2712/kitchen_sink-fff47adc7bb8b6f1f5820728ac9189c0e0aa9672.md) | 2,094 |  2,579,903 |  431 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/fff47adc7bb8b6f1f5820728ac9189c0e0aa9672

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24592836275)
