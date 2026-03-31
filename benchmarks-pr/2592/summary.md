| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-e54e1d46be57c7505278255496101506dc4ba372.md) | 3,816 |  12,000,265 |  938 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-e54e1d46be57c7505278255496101506dc4ba372.md) | 18,398 |  18,655,329 |  3,284 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-e54e1d46be57c7505278255496101506dc4ba372.md) | 1,418 |  4,137,067 |  375 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-e54e1d46be57c7505278255496101506dc4ba372.md) | 647 |  123,583 |  268 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-e54e1d46be57c7505278255496101506dc4ba372.md) | 911 |  1,745,757 |  282 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-e54e1d46be57c7505278255496101506dc4ba372.md) | 2,292 |  2,579,903 |  444 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/e54e1d46be57c7505278255496101506dc4ba372

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23816948993)
