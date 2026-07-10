| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/fibonacci-8b43b7616d136f7997a73fc29ff863900720903e.md) | 868 |  4,000,051 |  392 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/keccak-8b43b7616d136f7997a73fc29ff863900720903e.md) | 15,306 |  14,365,133 |  3,005 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/sha2_bench-8b43b7616d136f7997a73fc29ff863900720903e.md) | 7,978 |  11,167,961 |  1,002 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/regex-8b43b7616d136f7997a73fc29ff863900720903e.md) | 1,036 |  4,090,656 |  356 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/ecrecover-8b43b7616d136f7997a73fc29ff863900720903e.md) | 270 |  112,210 |  282 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/pairing-8b43b7616d136f7997a73fc29ff863900720903e.md) | 367 |  592,827 |  293 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2984/kitchen_sink-8b43b7616d136f7997a73fc29ff863900720903e.md) | 3,829 |  1,979,971 |  867 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/8b43b7616d136f7997a73fc29ff863900720903e

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29041314114)
