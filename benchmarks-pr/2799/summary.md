| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2799/fibonacci-393536d27a736e70b277e695e59d16c46fa1422c.md) | 3,749 |  12,000,265 |  915 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2799/keccak-393536d27a736e70b277e695e59d16c46fa1422c.md) | 18,489 |  18,655,329 |  3,256 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2799/sha2_bench-393536d27a736e70b277e695e59d16c46fa1422c.md) | 10,295 |  14,793,960 |  1,472 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2799/regex-393536d27a736e70b277e695e59d16c46fa1422c.md) | 1,402 |  4,137,067 |  356 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2799/ecrecover-393536d27a736e70b277e695e59d16c46fa1422c.md) | 602 |  123,583 |  256 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2799/pairing-393536d27a736e70b277e695e59d16c46fa1422c.md) | 904 |  1,745,757 |  270 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2799/kitchen_sink-393536d27a736e70b277e695e59d16c46fa1422c.md) | 1,890 |  2,579,903 |  408 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/393536d27a736e70b277e695e59d16c46fa1422c

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26188147561)
