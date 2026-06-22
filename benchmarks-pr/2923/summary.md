| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2923/fibonacci-42afe481ecd5e1577d8eb4d9dfa8312a2307a45e.md) | 1,029 |  4,000,051 |  396 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2923/keccak-42afe481ecd5e1577d8eb4d9dfa8312a2307a45e.md) | 16,176 |  14,365,133 |  3,004 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2923/sha2_bench-42afe481ecd5e1577d8eb4d9dfa8312a2307a45e.md) | 8,348 |  11,167,961 |  1,015 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2923/regex-42afe481ecd5e1577d8eb4d9dfa8312a2307a45e.md) | 1,224 |  4,090,656 |  360 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2923/ecrecover-42afe481ecd5e1577d8eb4d9dfa8312a2307a45e.md) | 434 |  112,210 |  279 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2923/pairing-42afe481ecd5e1577d8eb4d9dfa8312a2307a45e.md) | 601 |  592,827 |  301 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2923/kitchen_sink-42afe481ecd5e1577d8eb4d9dfa8312a2307a45e.md) | 3,906 |  1,979,971 |  869 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/42afe481ecd5e1577d8eb4d9dfa8312a2307a45e

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/27978715358)
