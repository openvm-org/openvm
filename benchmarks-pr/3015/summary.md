| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3015/fibonacci-d060ea1ed9c597409ffa22d0e20502100239a96b.md) | 463 |  4,000,051 |  230 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3015/keccak-d060ea1ed9c597409ffa22d0e20502100239a96b.md) | 8,528 |  14,365,133 |  1,521 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3015/sha2_bench-d060ea1ed9c597409ffa22d0e20502100239a96b.md) | 4,092 |  11,167,961 |  529 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3015/regex-d060ea1ed9c597409ffa22d0e20502100239a96b.md) | 566 |  4,090,656 |  218 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3015/ecrecover-d060ea1ed9c597409ffa22d0e20502100239a96b.md) | 220 |  112,210 |  183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3015/pairing-d060ea1ed9c597409ffa22d0e20502100239a96b.md) | 283 |  592,827 |  186 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3015/kitchen_sink-d060ea1ed9c597409ffa22d0e20502100239a96b.md) | 1,956 |  1,979,971 |  466 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3015/fibonacci_e2e-d060ea1ed9c597409ffa22d0e20502100239a96b.md) | 486 |  4,000,051 |  218 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3015/regex_e2e-d060ea1ed9c597409ffa22d0e20502100239a96b.md) | 664 |  4,090,656 |  213 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3015/ecrecover_e2e-d060ea1ed9c597409ffa22d0e20502100239a96b.md) | 216 |  112,210 |  174 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3015/pairing_e2e-d060ea1ed9c597409ffa22d0e20502100239a96b.md) | 312 |  592,827 |  174 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3015/kitchen_sink_e2e-d060ea1ed9c597409ffa22d0e20502100239a96b.md) | 2,297 |  1,979,971 |  455 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/d060ea1ed9c597409ffa22d0e20502100239a96b

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29350998229)
