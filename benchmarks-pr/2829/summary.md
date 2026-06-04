| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2829/fibonacci-36493a17993bf3ce4e802e80fd8c0a1683665b24.md) | 3,685 |  12,000,265 |  900 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2829/keccak-36493a17993bf3ce4e802e80fd8c0a1683665b24.md) | 18,072 |  18,655,329 |  3,297 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2829/sha2_bench-36493a17993bf3ce4e802e80fd8c0a1683665b24.md) | 9,822 |  14,793,960 |  1,434 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2829/regex-36493a17993bf3ce4e802e80fd8c0a1683665b24.md) | 1,395 |  4,137,067 |  354 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2829/ecrecover-36493a17993bf3ce4e802e80fd8c0a1683665b24.md) | 595 |  123,583 |  250 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2829/pairing-36493a17993bf3ce4e802e80fd8c0a1683665b24.md) | 888 |  1,745,757 |  264 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2829/kitchen_sink-36493a17993bf3ce4e802e80fd8c0a1683665b24.md) | 3,940 |  2,579,903 |  976 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/36493a17993bf3ce4e802e80fd8c0a1683665b24

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26976404117)
