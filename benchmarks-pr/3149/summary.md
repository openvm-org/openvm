| group | app.proof_time_ms | app.cycles | leaf.proof_time_ms |
| -- | -- | -- | -- |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3149/fibonacci-057e3c53522cc059db3ad0abaeb3bad9862017af.md) | 480 |  4,000,051 |  234 |
| [keccak](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3149/keccak-057e3c53522cc059db3ad0abaeb3bad9862017af.md) | 7,678 |  14,365,133 |  1,627 |
| [sha2_bench](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3149/sha2_bench-057e3c53522cc059db3ad0abaeb3bad9862017af.md) | 4,430 |  11,167,961 |  534 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3149/regex-057e3c53522cc059db3ad0abaeb3bad9862017af.md) | 747 |  4,090,656 |  217 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3149/ecrecover-057e3c53522cc059db3ad0abaeb3bad9862017af.md) | 209 |  112,210 |  187 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3149/pairing-057e3c53522cc059db3ad0abaeb3bad9862017af.md) | 250 |  592,827 |  174 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/3149/kitchen_sink-057e3c53522cc059db3ad0abaeb3bad9862017af.md) | 2,256 |  1,979,971 |  477 |

Note: cells_used metrics omitted because CUDA tracegen does not expose unpadded trace heights.


Commit: https://github.com/openvm-org/openvm/commit/057e3c53522cc059db3ad0abaeb3bad9862017af

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/33812815335)
