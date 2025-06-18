| group | app.proof_time_ms | app.cycles | app.cells_used | leaf.proof_time_ms | leaf.cycles | leaf.cells_used |
| -- | -- | -- | -- | -- | -- | -- |
| [verify_fibair](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/1734/verify_fibair-fe320786539a4af9a4f512ba879aef0e4063e4b6.md) |<span style='color: green'>(-72 [-6.0%])</span> 1,122 |  322,648 |  17,339,542 |- | - | - |
| [fibonacci](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/1734/fibonacci-fe320786539a4af9a4f512ba879aef0e4063e4b6.md) |<span style='color: red'>(+1671 [+62.1%])</span> 4,361 |  1,500,277 |  50,589,503 |<span style='color: green'>(-401 [-11.1%])</span> 3,223 |  1,248,002 |  69,833,666 |
| [regex](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/1734/regex-fe320786539a4af9a4f512ba879aef0e4063e4b6.md) |<span style='color: red'>(+2876 [+36.5%])</span> 10,754 |  4,165,432 |  166,519,456 |<span style='color: green'>(-4679 [-33.4%])</span> 9,319 | <span style='color: green'>(-602439 [-15.2%])</span> 3,348,989 | <span style='color: green'>(-74737651 [-24.6%])</span> 228,917,727 |
| [ecrecover](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/1734/ecrecover-fe320786539a4af9a4f512ba879aef0e4063e4b6.md) |<span style='color: red'>(+2194 [+188.3%])</span> 3,359 | <span style='color: red'>(+1042 [+0.8%])</span> 137,260 | <span style='color: red'>(+60427 [+0.7%])</span> 8,150,689 |<span style='color: green'>(-1137 [-9.7%])</span> 10,570 | <span style='color: green'>(-77687 [-2.6%])</span> 2,934,831 | <span style='color: green'>(-3205574 [-1.3%])</span> 241,887,734 |
| [pairing](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/1734/pairing-fe320786539a4af9a4f512ba879aef0e4063e4b6.md) |<span style='color: red'>(+1898 [+42.3%])</span> 6,384 |  1,862,964 | <span style='color: red'>(+230956 [+0.2%])</span> 97,508,739 |<span style='color: green'>(-4055 [-46.4%])</span> 4,683 | <span style='color: green'>(-564096 [-21.9%])</span> 2,010,455 | <span style='color: green'>(-70715097 [-34.4%])</span> 134,810,501 |
| [fib_e2e](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/1734/fib_e2e-fe320786539a4af9a4f512ba879aef0e4063e4b6.md) | 30,629 |  12,000,277 |  404,297,545 | 17,904 |  7,596,524 |  428,973,672 |
| [kitchen_sink](https://github.com/openvm-org/openvm/blob/benchmark-results/benchmarks-pr/1734/kitchen_sink-fe320786539a4af9a4f512ba879aef0e4063e4b6.md) | 17,629 |  154,595 |  898,132,732 | 24,455 |  7,991,137 |  732,640,405 |


Commit: https://github.com/openvm-org/openvm/commit/fe320786539a4af9a4f512ba879aef0e4063e4b6

[Benchmark Workflow](https://github.com/openvm-org/openvm/actions/runs/15741395256)
