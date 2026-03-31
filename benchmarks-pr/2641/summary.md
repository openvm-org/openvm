| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2641/fibonacci-793793ecc2ca14db2bc5acaac470a1d7452e93b7.md) | 3,831 |  12,000,265 |  950 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2641/keccak-793793ecc2ca14db2bc5acaac470a1d7452e93b7.md) | 15,715 |  1,235,218 |  2,201 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2641/regex-793793ecc2ca14db2bc5acaac470a1d7452e93b7.md) | 1,425 |  4,136,694 |  373 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2641/ecrecover-793793ecc2ca14db2bc5acaac470a1d7452e93b7.md) | 633 |  122,348 |  264 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2641/pairing-793793ecc2ca14db2bc5acaac470a1d7452e93b7.md) | 929 |  1,745,757 |  283 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2641/kitchen_sink-793793ecc2ca14db2bc5acaac470a1d7452e93b7.md) | 2,377 |  154,763 |  421 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/793793ecc2ca14db2bc5acaac470a1d7452e93b7

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23822948278)
