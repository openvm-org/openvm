| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3155/fibonacci-247cd09c441a76b1d1c6c12e64286049057230a6.md) | 474 |  4,000,051 |  232 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3155/keccak-247cd09c441a76b1d1c6c12e64286049057230a6.md) | 7,721 |  14,365,133 |  1,635 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3155/sha2_bench-247cd09c441a76b1d1c6c12e64286049057230a6.md) | 4,348 |  11,167,961 |  537 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3155/regex-247cd09c441a76b1d1c6c12e64286049057230a6.md) | 737 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3155/ecrecover-247cd09c441a76b1d1c6c12e64286049057230a6.md) | 209 |  112,210 |  186 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3155/pairing-247cd09c441a76b1d1c6c12e64286049057230a6.md) | 253 |  592,827 |  175 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3155/kitchen_sink-247cd09c441a76b1d1c6c12e64286049057230a6.md) | 2,236 |  1,979,971 |  472 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/247cd09c441a76b1d1c6c12e64286049057230a6

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/34603162238)
