| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/fibonacci-72efc2ce75481e768fffb6a7d4b04a1f651b0384.md) | 474 |  4,000,051 |  232 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/keccak-72efc2ce75481e768fffb6a7d4b04a1f651b0384.md) | 7,324 |  14,365,133 |  1,522 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/sha2_bench-72efc2ce75481e768fffb6a7d4b04a1f651b0384.md) | 4,149 |  11,167,961 |  523 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/regex-72efc2ce75481e768fffb6a7d4b04a1f651b0384.md) | 666 |  4,090,656 |  216 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/ecrecover-72efc2ce75481e768fffb6a7d4b04a1f651b0384.md) | 223 |  112,210 |  187 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/pairing-72efc2ce75481e768fffb6a7d4b04a1f651b0384.md) | 231 |  592,827 |  183 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/kitchen_sink-72efc2ce75481e768fffb6a7d4b04a1f651b0384.md) | 2,003 |  1,979,971 |  454 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/72efc2ce75481e768fffb6a7d4b04a1f651b0384

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31163387063)
