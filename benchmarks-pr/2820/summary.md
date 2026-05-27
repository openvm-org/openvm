| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2820/fibonacci-ee46ebc2779ec7becc23160e82383ce9d885bfa3.md) | 3,792 |  12,000,265 |  928 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2820/keccak-ee46ebc2779ec7becc23160e82383ce9d885bfa3.md) | 18,870 |  18,655,329 |  3,333 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2820/sha2_bench-ee46ebc2779ec7becc23160e82383ce9d885bfa3.md) | 10,250 |  14,793,960 |  1,474 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2820/regex-ee46ebc2779ec7becc23160e82383ce9d885bfa3.md) | 1,410 |  4,137,067 |  359 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2820/ecrecover-ee46ebc2779ec7becc23160e82383ce9d885bfa3.md) | 599 |  123,583 |  250 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2820/pairing-ee46ebc2779ec7becc23160e82383ce9d885bfa3.md) | 897 |  1,745,757 |  265 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2820/kitchen_sink-ee46ebc2779ec7becc23160e82383ce9d885bfa3.md) | 1,906 |  2,579,903 |  414 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/ee46ebc2779ec7becc23160e82383ce9d885bfa3

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26519914031)
