| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2636/fibonacci-ac3cc3e6a0e8c1ba76a439b9cd226490a83dcd36.md) | 3,860 |  12,000,265 |  952 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2636/keccak-ac3cc3e6a0e8c1ba76a439b9cd226490a83dcd36.md) | 15,640 |  1,235,218 |  2,188 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2636/regex-ac3cc3e6a0e8c1ba76a439b9cd226490a83dcd36.md) | 1,426 |  4,136,694 |  370 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2636/ecrecover-ac3cc3e6a0e8c1ba76a439b9cd226490a83dcd36.md) | 631 |  122,348 |  268 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2636/pairing-ac3cc3e6a0e8c1ba76a439b9cd226490a83dcd36.md) | 911 |  1,745,757 |  276 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2636/kitchen_sink-ac3cc3e6a0e8c1ba76a439b9cd226490a83dcd36.md) | 2,380 |  154,763 |  417 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/ac3cc3e6a0e8c1ba76a439b9cd226490a83dcd36

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23815042139)
