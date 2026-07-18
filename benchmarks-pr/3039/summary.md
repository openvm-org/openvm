| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3039/fibonacci-cee3c82b19fa70b2973085abe15b3e4862b3f377.md) | 412 |  4,000,051 |  232 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3039/keccak-cee3c82b19fa70b2973085abe15b3e4862b3f377.md) | 8,603 |  14,365,133 |  1,517 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3039/sha2_bench-cee3c82b19fa70b2973085abe15b3e4862b3f377.md) | 4,220 |  11,167,961 |  523 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3039/regex-cee3c82b19fa70b2973085abe15b3e4862b3f377.md) | 576 |  4,090,656 |  215 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3039/ecrecover-cee3c82b19fa70b2973085abe15b3e4862b3f377.md) | 218 |  112,210 |  181 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3039/pairing-cee3c82b19fa70b2973085abe15b3e4862b3f377.md) | 293 |  592,827 |  186 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3039/kitchen_sink-cee3c82b19fa70b2973085abe15b3e4862b3f377.md) | 1,910 |  1,979,971 |  461 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/cee3c82b19fa70b2973085abe15b3e4862b3f377

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29651449858)
