| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/fibonacci-69e291954323621b1e1e0faa4364e8f559fbcd21.md) | 417 |  4,000,051 |  227 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/keccak-69e291954323621b1e1e0faa4364e8f559fbcd21.md) | 8,360 |  14,365,133 |  1,522 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/sha2_bench-69e291954323621b1e1e0faa4364e8f559fbcd21.md) | 4,142 |  11,167,961 |  530 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/regex-69e291954323621b1e1e0faa4364e8f559fbcd21.md) | 501 |  4,090,656 |  189 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/ecrecover-69e291954323621b1e1e0faa4364e8f559fbcd21.md) | 220 |  112,210 |  185 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/pairing-69e291954323621b1e1e0faa4364e8f559fbcd21.md) | 265 |  592,827 |  183 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/2973/kitchen_sink-69e291954323621b1e1e0faa4364e8f559fbcd21.md) | 1,973 |  1,979,971 |  456 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/69e291954323621b1e1e0faa4364e8f559fbcd21

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/29454774823)
