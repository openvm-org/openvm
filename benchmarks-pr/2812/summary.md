| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2812/fibonacci-21987f7ca260b487ecdf03cb295e5a9e194f0a27.md) | 3,735 |  12,000,265 |  917 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2812/keccak-21987f7ca260b487ecdf03cb295e5a9e194f0a27.md) | 18,495 |  18,655,329 |  3,277 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2812/sha2_bench-21987f7ca260b487ecdf03cb295e5a9e194f0a27.md) | 10,169 |  14,793,960 |  1,457 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2812/regex-21987f7ca260b487ecdf03cb295e5a9e194f0a27.md) | 1,385 |  4,137,067 |  351 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2812/ecrecover-21987f7ca260b487ecdf03cb295e5a9e194f0a27.md) | 602 |  123,583 |  258 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2812/pairing-21987f7ca260b487ecdf03cb295e5a9e194f0a27.md) | 900 |  1,745,757 |  259 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2812/kitchen_sink-21987f7ca260b487ecdf03cb295e5a9e194f0a27.md) | 1,891 |  2,579,903 |  409 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/21987f7ca260b487ecdf03cb295e5a9e194f0a27

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/26339508402)
