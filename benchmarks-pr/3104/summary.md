| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/fibonacci-6106bd03ae0157fa0d68995b88e2fe5b69c3217d.md) | 458 |  4,000,051 |  229 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/keccak-6106bd03ae0157fa0d68995b88e2fe5b69c3217d.md) | 7,207 |  14,365,133 |  1,588 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/sha2_bench-6106bd03ae0157fa0d68995b88e2fe5b69c3217d.md) | 4,036 |  11,167,961 |  519 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/regex-6106bd03ae0157fa0d68995b88e2fe5b69c3217d.md) | 750 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/ecrecover-6106bd03ae0157fa0d68995b88e2fe5b69c3217d.md) | 206 |  112,210 |  185 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/pairing-6106bd03ae0157fa0d68995b88e2fe5b69c3217d.md) | 235 |  592,827 |  184 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/kitchen_sink-6106bd03ae0157fa0d68995b88e2fe5b69c3217d.md) | 2,130 |  1,979,971 |  457 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/6106bd03ae0157fa0d68995b88e2fe5b69c3217d

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/32430098388)
