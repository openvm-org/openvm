| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2954/fibonacci-7802fde0cffab92529fcf7e058b64cd8bee31fd7.md) | 3,063 |  12,000,265 |  678 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2954/keccak-7802fde0cffab92529fcf7e058b64cd8bee31fd7.md) | 16,519 |  18,655,329 |  3,071 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2954/sha2_bench-7802fde0cffab92529fcf7e058b64cd8bee31fd7.md) | 9,345 |  14,793,960 |  1,138 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2954/regex-7802fde0cffab92529fcf7e058b64cd8bee31fd7.md) | 1,164 |  4,137,067 |  352 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2954/ecrecover-7802fde0cffab92529fcf7e058b64cd8bee31fd7.md) | 600 |  123,583 |  283 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2954/pairing-7802fde0cffab92529fcf7e058b64cd8bee31fd7.md) | 935 |  1,745,757 |  306 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2954/kitchen_sink-7802fde0cffab92529fcf7e058b64cd8bee31fd7.md) | 4,146 |  2,579,903 |  889 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/7802fde0cffab92529fcf7e058b64cd8bee31fd7

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28487568806)
