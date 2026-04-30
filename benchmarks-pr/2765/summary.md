| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/fibonacci-aafca0a55428b280f194b48c3276bfb64e22777a.md) | 1,903 |  4,000,051 |  539 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/keccak-aafca0a55428b280f194b48c3276bfb64e22777a.md) | 13,589 |  14,365,133 |  2,238 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/sha2_bench-aafca0a55428b280f194b48c3276bfb64e22777a.md) | 9,429 |  11,167,961 |  1,263 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/regex-aafca0a55428b280f194b48c3276bfb64e22777a.md) | 1,575 |  4,090,656 |  374 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/ecrecover-aafca0a55428b280f194b48c3276bfb64e22777a.md) | 636 |  112,210 |  288 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/pairing-aafca0a55428b280f194b48c3276bfb64e22777a.md) | 752 |  592,827 |  279 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2765/kitchen_sink-aafca0a55428b280f194b48c3276bfb64e22777a.md) | 2,069 |  1,979,971 |  434 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/aafca0a55428b280f194b48c3276bfb64e22777a

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/25182366557)
