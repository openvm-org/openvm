| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2843/fibonacci-36493a17993bf3ce4e802e80fd8c0a1683665b24.md) | 3,784 |  12,000,265 |  926 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2843/keccak-36493a17993bf3ce4e802e80fd8c0a1683665b24.md) | 18,157 |  18,655,329 |  3,291 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2843/sha2_bench-36493a17993bf3ce4e802e80fd8c0a1683665b24.md) | 9,982 |  14,793,960 |  1,444 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2843/regex-36493a17993bf3ce4e802e80fd8c0a1683665b24.md) | 1,383 |  4,137,067 |  354 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2843/ecrecover-36493a17993bf3ce4e802e80fd8c0a1683665b24.md) | 600 |  123,583 |  254 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2843/pairing-36493a17993bf3ce4e802e80fd8c0a1683665b24.md) | 890 |  1,745,757 |  268 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2843/kitchen_sink-36493a17993bf3ce4e802e80fd8c0a1683665b24.md) | 3,837 |  2,579,903 |  947 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/36493a17993bf3ce4e802e80fd8c0a1683665b24

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26976761694)
