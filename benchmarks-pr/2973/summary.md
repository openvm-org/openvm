| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/fibonacci-3f50d2208aa804c3610626d092f13da25e7ac0a9.md) | 409 |  4,000,051 |  226 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/keccak-3f50d2208aa804c3610626d092f13da25e7ac0a9.md) | 8,580 |  14,365,133 |  1,547 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/sha2_bench-3f50d2208aa804c3610626d092f13da25e7ac0a9.md) | 4,106 |  11,167,961 |  526 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/regex-3f50d2208aa804c3610626d092f13da25e7ac0a9.md) | 496 |  4,090,656 |  191 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/ecrecover-3f50d2208aa804c3610626d092f13da25e7ac0a9.md) | 220 |  112,210 |  183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/pairing-3f50d2208aa804c3610626d092f13da25e7ac0a9.md) | 258 |  592,827 |  184 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/kitchen_sink-3f50d2208aa804c3610626d092f13da25e7ac0a9.md) | 1,978 |  1,979,971 |  455 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/3f50d2208aa804c3610626d092f13da25e7ac0a9

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29433024989)
