| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3114/fibonacci-c977c46950162baf22d27f7d3a3ad7b649e523ff.md) | 480 |  4,000,051 |  230 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3114/keccak-c977c46950162baf22d27f7d3a3ad7b649e523ff.md) | 7,386 |  14,365,133 |  1,528 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3114/sha2_bench-c977c46950162baf22d27f7d3a3ad7b649e523ff.md) | 4,189 |  11,167,961 |  528 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3114/regex-c977c46950162baf22d27f7d3a3ad7b649e523ff.md) | 650 |  4,090,656 |  210 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3114/ecrecover-c977c46950162baf22d27f7d3a3ad7b649e523ff.md) | 227 |  112,210 |  183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3114/pairing-c977c46950162baf22d27f7d3a3ad7b649e523ff.md) | 236 |  592,827 |  181 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3114/kitchen_sink-c977c46950162baf22d27f7d3a3ad7b649e523ff.md) | 2,036 |  1,979,971 |  460 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/c977c46950162baf22d27f7d3a3ad7b649e523ff

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31507411143)
