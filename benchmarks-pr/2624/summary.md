| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2624/fibonacci-23779d207738a03a56c2e33d67668b58afd09d1a.md) | 3,846 |  12,000,265 |  942 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2624/keccak-23779d207738a03a56c2e33d67668b58afd09d1a.md) | 15,823 |  1,235,218 |  2,196 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2624/regex-23779d207738a03a56c2e33d67668b58afd09d1a.md) | 1,414 |  4,136,694 |  366 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2624/ecrecover-23779d207738a03a56c2e33d67668b58afd09d1a.md) | 636 |  122,348 |  268 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2624/pairing-23779d207738a03a56c2e33d67668b58afd09d1a.md) | 922 |  1,745,757 |  290 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2624/kitchen_sink-23779d207738a03a56c2e33d67668b58afd09d1a.md) | 2,371 |  154,763 |  408 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/23779d207738a03a56c2e33d67668b58afd09d1a

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23610594188)
