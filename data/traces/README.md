# Traces

Per-minute invocation counts of the 4 functions used in the paper.

`<fid>_minute.csv`: `minute` (0-20159, 14 days, day 1 = minutes 0-1439), `count_raw` (original Azure counts),
`count_scaled` (scaled down per minute counts to match testbed capacity),

| fid | trigger | scale ratio | raw total | scaled total |
|---|---|---|---|---|
| 20606 | http | 1/11 | 102644 | 9331 |
| 28518 | orchestration | 1 | 9744 | 9744 |
| 10623 | queue | 5/28 | 50911 | 9091 |
| 22217 | orchestration | 1 | 10910 | 10910 |