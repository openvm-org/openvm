| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2812/fibonacci-a801bd4512e9a4726eda7b6babf1a0476ee23524.md) | 3,771 |  12,000,265 |  927 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2812/keccak-a801bd4512e9a4726eda7b6babf1a0476ee23524.md) | 18,774 |  18,655,329 |  3,303 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2812/sha2_bench-a801bd4512e9a4726eda7b6babf1a0476ee23524.md) | 10,116 |  14,793,960 |  1,448 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2812/regex-a801bd4512e9a4726eda7b6babf1a0476ee23524.md) | 1,402 |  4,137,067 |  353 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2812/ecrecover-a801bd4512e9a4726eda7b6babf1a0476ee23524.md) | 604 |  123,583 |  252 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2812/pairing-a801bd4512e9a4726eda7b6babf1a0476ee23524.md) | 899 |  1,745,757 |  265 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2812/kitchen_sink-a801bd4512e9a4726eda7b6babf1a0476ee23524.md) | 1,894 |  2,579,903 |  415 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/a801bd4512e9a4726eda7b6babf1a0476ee23524

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26339126660)
