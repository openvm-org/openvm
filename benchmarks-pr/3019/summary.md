| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/fibonacci-e25813b38eb4376172e3424155eed7bd3f598d4a.md) | 473 |  4,000,051 |  231 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/keccak-e25813b38eb4376172e3424155eed7bd3f598d4a.md) | 8,745 |  14,365,133 |  1,517 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/sha2_bench-e25813b38eb4376172e3424155eed7bd3f598d4a.md) | 3,910 |  11,167,961 |  514 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/regex-e25813b38eb4376172e3424155eed7bd3f598d4a.md) | 501 |  4,090,656 |  188 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/ecrecover-e25813b38eb4376172e3424155eed7bd3f598d4a.md) | 219 |  112,210 |  184 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/pairing-e25813b38eb4376172e3424155eed7bd3f598d4a.md) | 276 |  592,827 |  185 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/kitchen_sink-e25813b38eb4376172e3424155eed7bd3f598d4a.md) | 1,930 |  1,979,971 |  468 |
| [fibonacci_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/fibonacci_e2e-e25813b38eb4376172e3424155eed7bd3f598d4a.md) | 487 |  4,000,051 |  218 |
| [regex_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/regex_e2e-e25813b38eb4376172e3424155eed7bd3f598d4a.md) | 586 |  4,090,656 |  183 |
| [ecrecover_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/ecrecover_e2e-e25813b38eb4376172e3424155eed7bd3f598d4a.md) | 215 |  112,210 |  172 |
| [pairing_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/pairing_e2e-e25813b38eb4376172e3424155eed7bd3f598d4a.md) | 292 |  592,827 |  178 |
| [kitchen_sink_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3019/kitchen_sink_e2e-e25813b38eb4376172e3424155eed7bd3f598d4a.md) | 2,260 |  1,979,971 |  454 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/e25813b38eb4376172e3424155eed7bd3f598d4a

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29409829690)
