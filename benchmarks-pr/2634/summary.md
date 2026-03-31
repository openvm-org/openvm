| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2634/fibonacci-10a3bf33d2eec42d1b554dbfe35aa1ca55bb449c.md) | 3,812 |  12,000,265 |  942 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2634/keccak-10a3bf33d2eec42d1b554dbfe35aa1ca55bb449c.md) | 18,603 |  18,655,329 |  3,314 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2634/regex-10a3bf33d2eec42d1b554dbfe35aa1ca55bb449c.md) | 1,428 |  4,137,067 |  374 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2634/ecrecover-10a3bf33d2eec42d1b554dbfe35aa1ca55bb449c.md) | 648 |  123,583 |  268 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2634/pairing-10a3bf33d2eec42d1b554dbfe35aa1ca55bb449c.md) | 902 |  1,745,757 |  277 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2634/kitchen_sink-10a3bf33d2eec42d1b554dbfe35aa1ca55bb449c.md) | 2,283 |  2,579,903 |  440 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/10a3bf33d2eec42d1b554dbfe35aa1ca55bb449c

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23810822553)
