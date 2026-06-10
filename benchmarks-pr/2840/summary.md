| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2840/fibonacci-0e6d6a6391038955d1371c8fa551e426de5782c4.md) | 1,523 |  4,000,051 |  530 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2840/keccak-0e6d6a6391038955d1371c8fa551e426de5782c4.md) | 16,387 |  14,365,133 |  3,047 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2840/sha2_bench-0e6d6a6391038955d1371c8fa551e426de5782c4.md) | 10,518 |  11,167,961 |  1,958 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2840/regex-0e6d6a6391038955d1371c8fa551e426de5782c4.md) | 1,520 |  4,090,656 |  436 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2840/ecrecover-0e6d6a6391038955d1371c8fa551e426de5782c4.md) | 447 |  112,210 |  311 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2840/pairing-0e6d6a6391038955d1371c8fa551e426de5782c4.md) | 602 |  592,827 |  295 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2840/kitchen_sink-0e6d6a6391038955d1371c8fa551e426de5782c4.md) | 3,920 |  1,979,971 |  862 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/0e6d6a6391038955d1371c8fa551e426de5782c4

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27309603769)
