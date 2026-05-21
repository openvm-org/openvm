| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2802/fibonacci-db27775bcd5b02e59886b78fbceed2744654b62b.md) | 1,566 |  4,000,051 |  431 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2802/keccak-db27775bcd5b02e59886b78fbceed2744654b62b.md) | 13,914 |  14,365,133 |  2,209 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2802/sha2_bench-db27775bcd5b02e59886b78fbceed2744654b62b.md) | 9,084 |  11,167,961 |  1,406 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2802/regex-db27775bcd5b02e59886b78fbceed2744654b62b.md) | 1,503 |  4,090,656 |  361 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2802/ecrecover-db27775bcd5b02e59886b78fbceed2744654b62b.md) | 479 |  112,210 |  275 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2802/pairing-db27775bcd5b02e59886b78fbceed2744654b62b.md) | 602 |  592,827 |  258 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2802/kitchen_sink-db27775bcd5b02e59886b78fbceed2744654b62b.md) | 2,218 |  1,979,971 |  411 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/db27775bcd5b02e59886b78fbceed2744654b62b

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26236088499)
