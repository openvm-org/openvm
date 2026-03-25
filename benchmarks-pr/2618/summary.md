| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2618/fibonacci-ca22164eb8fe6671b835f458f1c665881788cbda.md) | 4,169 |  12,000,265 |  1,360 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2618/keccak-ca22164eb8fe6671b835f458f1c665881788cbda.md) | 19,174 |  1,235,218 |  3,358 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2618/regex-ca22164eb8fe6671b835f458f1c665881788cbda.md) | 1,610 |  4,136,694 |  534 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2618/ecrecover-ca22164eb8fe6671b835f458f1c665881788cbda.md) | 647 |  122,348 |  336 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2618/pairing-ca22164eb8fe6671b835f458f1c665881788cbda.md) | 1,053 |  1,745,757 |  345 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2618/kitchen_sink-ca22164eb8fe6671b835f458f1c665881788cbda.md) | 3,307 |  154,763 |  724 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/ca22164eb8fe6671b835f458f1c665881788cbda

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23559215309)
