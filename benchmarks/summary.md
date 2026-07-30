| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/fibonacci-b4f746802adf68edfa20e350c0c6a8fb03495490.md) | 1,583 |  12,000,265 |  360 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/keccak-b4f746802adf68edfa20e350c0c6a8fb03495490.md) | 9,255 |  18,655,329 |  1,520 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/sha2_bench-b4f746802adf68edfa20e350c0c6a8fb03495490.md) | 4,916 |  14,793,960 |  575 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/regex-b4f746802adf68edfa20e350c0c6a8fb03495490.md) | 661 |  4,137,067 |  213 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/ecrecover-b4f746802adf68edfa20e350c0c6a8fb03495490.md) | 437 |  123,583 |  186 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/pairing-b4f746802adf68edfa20e350c0c6a8fb03495490.md) | 588 |  1,745,757 |  190 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/kitchen_sink-b4f746802adf68edfa20e350c0c6a8fb03495490.md) | 2,221 |  2,579,903 |  478 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/b4f746802adf68edfa20e350c0c6a8fb03495490

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30557739165)
