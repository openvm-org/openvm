| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2840/fibonacci-1886d1bedac849b29bc0ac24332b136f4f8161e3.md) | 1,402 |  4,000,051 |  431 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2840/keccak-1886d1bedac849b29bc0ac24332b136f4f8161e3.md) | 13,761 |  14,365,133 |  2,379 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2840/sha2_bench-1886d1bedac849b29bc0ac24332b136f4f8161e3.md) | 8,914 |  11,167,961 |  1,413 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2840/regex-1886d1bedac849b29bc0ac24332b136f4f8161e3.md) | 1,406 |  4,090,656 |  358 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2840/ecrecover-1886d1bedac849b29bc0ac24332b136f4f8161e3.md) | 432 |  112,210 |  272 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2840/pairing-1886d1bedac849b29bc0ac24332b136f4f8161e3.md) | 570 |  592,827 |  255 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2840/kitchen_sink-1886d1bedac849b29bc0ac24332b136f4f8161e3.md) | 3,691 |  1,979,971 |  930 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/1886d1bedac849b29bc0ac24332b136f4f8161e3

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26979266967)
