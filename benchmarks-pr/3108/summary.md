| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/fibonacci-cbd1fd2ca2451d6869a91a23bd2947888a99ee60.md) | 469 |  4,000,051 |  232 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/keccak-cbd1fd2ca2451d6869a91a23bd2947888a99ee60.md) | 7,343 |  14,365,133 |  1,591 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/sha2_bench-cbd1fd2ca2451d6869a91a23bd2947888a99ee60.md) | 4,033 |  11,167,961 |  512 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/regex-cbd1fd2ca2451d6869a91a23bd2947888a99ee60.md) | 725 |  4,090,656 |  217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/ecrecover-cbd1fd2ca2451d6869a91a23bd2947888a99ee60.md) | 204 |  112,210 |  181 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/pairing-cbd1fd2ca2451d6869a91a23bd2947888a99ee60.md) | 245 |  592,827 |  169 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3108/kitchen_sink-cbd1fd2ca2451d6869a91a23bd2947888a99ee60.md) | 2,164 |  1,979,971 |  459 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/cbd1fd2ca2451d6869a91a23bd2947888a99ee60

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/33114032786)
