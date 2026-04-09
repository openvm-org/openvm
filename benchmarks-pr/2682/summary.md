| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2682/fibonacci-c135323079ec2dfe770ed31616022a31bafdf0da.md) | 3,862 |  12,000,265 |  962 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2682/keccak-c135323079ec2dfe770ed31616022a31bafdf0da.md) | 18,724 |  18,655,329 |  3,343 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2682/regex-c135323079ec2dfe770ed31616022a31bafdf0da.md) | 1,411 |  4,137,067 |  372 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2682/ecrecover-c135323079ec2dfe770ed31616022a31bafdf0da.md) | 644 |  123,583 |  269 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2682/pairing-c135323079ec2dfe770ed31616022a31bafdf0da.md) | 907 |  1,745,757 |  285 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2682/kitchen_sink-c135323079ec2dfe770ed31616022a31bafdf0da.md) | 2,142 |  2,579,903 |  433 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/c135323079ec2dfe770ed31616022a31bafdf0da

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24189102735)
