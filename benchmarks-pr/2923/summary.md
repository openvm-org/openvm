| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2923/fibonacci-bd9a7b6040e5bb45775a15e3a56686a720239296.md) | 1,024 |  4,000,051 |  386 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2923/keccak-bd9a7b6040e5bb45775a15e3a56686a720239296.md) | 16,434 |  14,365,133 |  3,028 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2923/sha2_bench-bd9a7b6040e5bb45775a15e3a56686a720239296.md) | 8,129 |  11,167,961 |  999 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2923/regex-bd9a7b6040e5bb45775a15e3a56686a720239296.md) | 1,209 |  4,090,656 |  352 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2923/ecrecover-bd9a7b6040e5bb45775a15e3a56686a720239296.md) | 438 |  112,210 |  282 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2923/pairing-bd9a7b6040e5bb45775a15e3a56686a720239296.md) | 599 |  592,827 |  300 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2923/kitchen_sink-bd9a7b6040e5bb45775a15e3a56686a720239296.md) | 3,882 |  1,979,971 |  856 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/bd9a7b6040e5bb45775a15e3a56686a720239296

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28057568622)
