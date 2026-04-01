| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/fibonacci-2dc3ff8dba63cf5039f7937e24bfbeb27428fa4d.md) | 3,814 |  12,000,265 |  938 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/keccak-2dc3ff8dba63cf5039f7937e24bfbeb27428fa4d.md) | 18,490 |  18,655,329 |  3,303 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/regex-2dc3ff8dba63cf5039f7937e24bfbeb27428fa4d.md) | 1,432 |  4,137,067 |  374 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/ecrecover-2dc3ff8dba63cf5039f7937e24bfbeb27428fa4d.md) | 640 |  123,583 |  272 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/pairing-2dc3ff8dba63cf5039f7937e24bfbeb27428fa4d.md) | 909 |  1,745,757 |  289 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2592/kitchen_sink-2dc3ff8dba63cf5039f7937e24bfbeb27428fa4d.md) | 2,274 |  2,579,903 |  441 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/2dc3ff8dba63cf5039f7937e24bfbeb27428fa4d

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/23875908679)
