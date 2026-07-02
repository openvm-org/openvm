| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2955/fibonacci-deb7543545d677e040e9b42bbe35b59f1c8dc78f.md) | 3,073 |  12,000,265 |  677 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2955/keccak-deb7543545d677e040e9b42bbe35b59f1c8dc78f.md) | 16,678 |  18,655,329 |  3,093 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2955/sha2_bench-deb7543545d677e040e9b42bbe35b59f1c8dc78f.md) | 9,250 |  14,793,960 |  1,134 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2955/regex-deb7543545d677e040e9b42bbe35b59f1c8dc78f.md) | 1,169 |  4,137,067 |  353 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2955/ecrecover-deb7543545d677e040e9b42bbe35b59f1c8dc78f.md) | 601 |  123,583 |  288 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2955/pairing-deb7543545d677e040e9b42bbe35b59f1c8dc78f.md) | 935 |  1,745,757 |  307 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2955/kitchen_sink-deb7543545d677e040e9b42bbe35b59f1c8dc78f.md) | 4,186 |  2,579,903 |  902 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/deb7543545d677e040e9b42bbe35b59f1c8dc78f

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/28621750752)
