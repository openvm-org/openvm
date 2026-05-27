| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2822/fibonacci-93a1cb39b10da30138cdba586a73007fb45e9182.md) | 3,804 |  12,000,265 |  923 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2822/keccak-93a1cb39b10da30138cdba586a73007fb45e9182.md) | 18,899 |  18,655,329 |  3,324 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2822/sha2_bench-93a1cb39b10da30138cdba586a73007fb45e9182.md) | 10,404 |  14,793,960 |  1,503 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2822/regex-93a1cb39b10da30138cdba586a73007fb45e9182.md) | 1,398 |  4,137,067 |  355 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2822/ecrecover-93a1cb39b10da30138cdba586a73007fb45e9182.md) | 622 |  123,583 |  259 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2822/pairing-93a1cb39b10da30138cdba586a73007fb45e9182.md) | 907 |  1,745,757 |  265 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2822/kitchen_sink-93a1cb39b10da30138cdba586a73007fb45e9182.md) | 1,908 |  2,579,903 |  413 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/93a1cb39b10da30138cdba586a73007fb45e9182

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26523307862)
