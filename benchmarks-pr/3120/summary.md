| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3120/fibonacci-ec7cb9c3272e9fec9aeee23432127eec0179fd48.md) | 453 |  4,000,051 |  229 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3120/keccak-ec7cb9c3272e9fec9aeee23432127eec0179fd48.md) | 7,191 |  14,365,133 |  1,627 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3120/sha2_bench-ec7cb9c3272e9fec9aeee23432127eec0179fd48.md) | 4,198 |  11,167,961 |  523 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3120/regex-ec7cb9c3272e9fec9aeee23432127eec0179fd48.md) | 713 |  4,090,656 |  228 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3120/ecrecover-ec7cb9c3272e9fec9aeee23432127eec0179fd48.md) | 209 |  112,210 |  185 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3120/pairing-ec7cb9c3272e9fec9aeee23432127eec0179fd48.md) | 234 |  592,827 |  184 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3120/kitchen_sink-ec7cb9c3272e9fec9aeee23432127eec0179fd48.md) | 2,164 |  1,979,971 |  462 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/ec7cb9c3272e9fec9aeee23432127eec0179fd48

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31752739045)
