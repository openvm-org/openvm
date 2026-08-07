| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/fibonacci-efb2bad6dc6ec643587843a014cadc837ad03fcd.md) | 476 |  4,000,051 |  228 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/keccak-efb2bad6dc6ec643587843a014cadc837ad03fcd.md) | 7,458 |  14,365,133 |  1,534 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/sha2_bench-efb2bad6dc6ec643587843a014cadc837ad03fcd.md) | 4,163 |  11,167,961 |  523 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/regex-efb2bad6dc6ec643587843a014cadc837ad03fcd.md) | 653 |  4,090,656 |  213 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/ecrecover-efb2bad6dc6ec643587843a014cadc837ad03fcd.md) | 220 |  112,210 |  180 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/pairing-efb2bad6dc6ec643587843a014cadc837ad03fcd.md) | 230 |  592,827 |  183 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3109/kitchen_sink-efb2bad6dc6ec643587843a014cadc837ad03fcd.md) | 2,048 |  1,979,971 |  463 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/efb2bad6dc6ec643587843a014cadc837ad03fcd

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/31164252800)
