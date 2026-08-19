| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/fibonacci-05fcffb7f19c34ba6b42c4ad27af8358d7ae937f.md) | 436 |  4,000,051 |  225 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/keccak-05fcffb7f19c34ba6b42c4ad27af8358d7ae937f.md) | 7,220 |  14,365,133 |  1,586 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/sha2_bench-05fcffb7f19c34ba6b42c4ad27af8358d7ae937f.md) | 4,099 |  11,167,961 |  515 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/regex-05fcffb7f19c34ba6b42c4ad27af8358d7ae937f.md) | 717 |  4,090,656 |  214 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/ecrecover-05fcffb7f19c34ba6b42c4ad27af8358d7ae937f.md) | 206 |  112,210 |  181 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/pairing-05fcffb7f19c34ba6b42c4ad27af8358d7ae937f.md) | 241 |  592,827 |  187 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3104/kitchen_sink-05fcffb7f19c34ba6b42c4ad27af8358d7ae937f.md) | 2,167 |  1,979,971 |  459 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/05fcffb7f19c34ba6b42c4ad27af8358d7ae937f

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/32312553842)
