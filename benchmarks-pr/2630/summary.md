| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2630/fibonacci-e55af66968149eb3bbb0714fe896ccb1abd424ad.md) | 3,773 |  12,000,265 |  928 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2630/keccak-e55af66968149eb3bbb0714fe896ccb1abd424ad.md) | 18,397 |  18,655,329 |  3,290 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2630/regex-e55af66968149eb3bbb0714fe896ccb1abd424ad.md) | 1,424 |  4,137,067 |  374 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2630/ecrecover-e55af66968149eb3bbb0714fe896ccb1abd424ad.md) | 644 |  123,583 |  269 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2630/pairing-e55af66968149eb3bbb0714fe896ccb1abd424ad.md) | 898 |  1,745,757 |  279 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2630/kitchen_sink-e55af66968149eb3bbb0714fe896ccb1abd424ad.md) | 2,287 |  2,579,903 |  443 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/e55af66968149eb3bbb0714fe896ccb1abd424ad

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23803255792)
