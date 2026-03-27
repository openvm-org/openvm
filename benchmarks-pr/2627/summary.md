| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2627/fibonacci-644c3ceed49888efcb19d6595e77ef3277b36eb4.md) | 3,832 |  12,000,265 |  938 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2627/keccak-644c3ceed49888efcb19d6595e77ef3277b36eb4.md) | 15,534 |  1,235,218 |  2,174 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2627/regex-644c3ceed49888efcb19d6595e77ef3277b36eb4.md) | 1,404 |  4,136,694 |  372 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2627/ecrecover-644c3ceed49888efcb19d6595e77ef3277b36eb4.md) | 645 |  122,348 |  269 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2627/pairing-644c3ceed49888efcb19d6595e77ef3277b36eb4.md) | 910 |  1,745,757 |  281 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2627/kitchen_sink-644c3ceed49888efcb19d6595e77ef3277b36eb4.md) | 2,400 |  154,763 |  417 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/644c3ceed49888efcb19d6595e77ef3277b36eb4

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23651788599)
