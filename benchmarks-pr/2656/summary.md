| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2656/fibonacci-2e9e7edf858160441fc642c42fdba8d490669a98.md) | 3,829 |  12,000,265 |  935 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2656/keccak-2e9e7edf858160441fc642c42fdba8d490669a98.md) | 15,661 |  1,235,218 |  2,196 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2656/regex-2e9e7edf858160441fc642c42fdba8d490669a98.md) | 1,409 |  4,136,694 |  370 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2656/ecrecover-2e9e7edf858160441fc642c42fdba8d490669a98.md) | 639 |  122,348 |  266 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2656/pairing-2e9e7edf858160441fc642c42fdba8d490669a98.md) | 924 |  1,745,757 |  280 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2656/kitchen_sink-2e9e7edf858160441fc642c42fdba8d490669a98.md) | 2,369 |  154,763 |  412 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/2e9e7edf858160441fc642c42fdba8d490669a98

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23925123813)
