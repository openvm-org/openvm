| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2684/fibonacci-b409121ddb03cc0bd95846462ca7ab8f27958ec7.md) | 3,773 |  12,000,265 |  936 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2684/keccak-b409121ddb03cc0bd95846462ca7ab8f27958ec7.md) | 18,480 |  18,655,329 |  3,328 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2684/regex-b409121ddb03cc0bd95846462ca7ab8f27958ec7.md) | 1,399 |  4,137,067 |  371 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2684/ecrecover-b409121ddb03cc0bd95846462ca7ab8f27958ec7.md) | 646 |  123,583 |  273 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2684/pairing-b409121ddb03cc0bd95846462ca7ab8f27958ec7.md) | 907 |  1,745,757 |  283 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2684/kitchen_sink-b409121ddb03cc0bd95846462ca7ab8f27958ec7.md) | 2,167 |  2,579,903 |  437 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/b409121ddb03cc0bd95846462ca7ab8f27958ec7

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24197974437)
