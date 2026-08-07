| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/fibonacci-b55f383310a34e08625d868a2b038fcbca1d3b19.md) | 441 |  4,000,051 |  230 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/keccak-b55f383310a34e08625d868a2b038fcbca1d3b19.md) | 7,140 |  14,365,133 |  1,590 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/sha2_bench-b55f383310a34e08625d868a2b038fcbca1d3b19.md) | 4,111 |  11,167,961 |  521 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/regex-b55f383310a34e08625d868a2b038fcbca1d3b19.md) | 709 |  4,090,656 |  217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/ecrecover-b55f383310a34e08625d868a2b038fcbca1d3b19.md) | 201 |  112,210 |  183 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/pairing-b55f383310a34e08625d868a2b038fcbca1d3b19.md) | 233 |  592,827 |  186 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/kitchen_sink-b55f383310a34e08625d868a2b038fcbca1d3b19.md) | 2,149 |  1,979,971 |  461 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/b55f383310a34e08625d868a2b038fcbca1d3b19

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31220685840)
