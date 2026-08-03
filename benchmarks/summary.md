| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/fibonacci-f6595e3a66e6fd5c105fe9fbf336c9733932ba77.md) | 1,591 |  12,000,265 |  359 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/keccak-f6595e3a66e6fd5c105fe9fbf336c9733932ba77.md) | 9,399 |  18,655,329 |  1,531 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/sha2_bench-f6595e3a66e6fd5c105fe9fbf336c9733932ba77.md) | 4,999 |  14,793,960 |  579 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/regex-f6595e3a66e6fd5c105fe9fbf336c9733932ba77.md) | 661 |  4,137,067 |  214 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/ecrecover-f6595e3a66e6fd5c105fe9fbf336c9733932ba77.md) | 424 |  123,583 |  183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/pairing-f6595e3a66e6fd5c105fe9fbf336c9733932ba77.md) | 553 |  1,745,757 |  189 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks/kitchen_sink-f6595e3a66e6fd5c105fe9fbf336c9733932ba77.md) | 2,233 |  2,579,903 |  482 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/f6595e3a66e6fd5c105fe9fbf336c9733932ba77

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/30815163915)
