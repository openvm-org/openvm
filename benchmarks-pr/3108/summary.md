| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/fibonacci-4c0f2d45ff3dc4599d52db21570db0e5860d92d5.md) | 461 |  4,000,051 |  233 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/keccak-4c0f2d45ff3dc4599d52db21570db0e5860d92d5.md) | 7,319 |  14,365,133 |  1,614 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/sha2_bench-4c0f2d45ff3dc4599d52db21570db0e5860d92d5.md) | 4,078 |  11,167,961 |  515 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/regex-4c0f2d45ff3dc4599d52db21570db0e5860d92d5.md) | 729 |  4,090,656 |  218 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/ecrecover-4c0f2d45ff3dc4599d52db21570db0e5860d92d5.md) | 207 |  112,210 |  189 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/pairing-4c0f2d45ff3dc4599d52db21570db0e5860d92d5.md) | 240 |  592,827 |  170 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/kitchen_sink-4c0f2d45ff3dc4599d52db21570db0e5860d92d5.md) | 2,153 |  1,979,971 |  458 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/4c0f2d45ff3dc4599d52db21570db0e5860d92d5

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/33119095923)
