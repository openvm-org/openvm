| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2627/fibonacci-3b10268960f63deed4a9ec95caa50a88f2813238.md) | 3,884 |  12,000,265 |  955 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2627/keccak-3b10268960f63deed4a9ec95caa50a88f2813238.md) | 15,689 |  1,235,218 |  2,202 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2627/regex-3b10268960f63deed4a9ec95caa50a88f2813238.md) | 1,414 |  4,136,694 |  365 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2627/ecrecover-3b10268960f63deed4a9ec95caa50a88f2813238.md) | 632 |  122,348 |  269 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2627/pairing-3b10268960f63deed4a9ec95caa50a88f2813238.md) | 923 |  1,745,757 |  278 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2627/kitchen_sink-3b10268960f63deed4a9ec95caa50a88f2813238.md) | 2,387 |  154,763 |  417 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/3b10268960f63deed4a9ec95caa50a88f2813238

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23857870520)
