| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2664/fibonacci-5913437537f1bd113928475b20b9d00c427a914d.md) | 3,838 |  12,000,265 |  942 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2664/keccak-5913437537f1bd113928475b20b9d00c427a914d.md) | 15,665 |  1,235,218 |  2,201 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2664/regex-5913437537f1bd113928475b20b9d00c427a914d.md) | 1,417 |  4,136,694 |  371 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2664/ecrecover-5913437537f1bd113928475b20b9d00c427a914d.md) | 635 |  122,348 |  267 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2664/pairing-5913437537f1bd113928475b20b9d00c427a914d.md) | 912 |  1,745,757 |  274 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2664/kitchen_sink-5913437537f1bd113928475b20b9d00c427a914d.md) | 2,397 |  154,763 |  418 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/5913437537f1bd113928475b20b9d00c427a914d

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/24047466578)
