# Forgetting check: ft5 vs base on MBPP+ (300 problems)

Criterion 3 (LOGBOOK, frozen 2026-10-22): the fine-tuned model may lose at most 3 points of pass@1 against the same base.

- reference arm A = `base` (base-qwen3.5-9b), candidate arm B = `ft5` (opencv-expert-q5)
- dataset evalplus/mbppplus @ b2d74c91837c3f2a20c1299ae98133cbe7cfa077; slice file sha256 4820ab7de00c34df1f419ac26c73ba97beff8984c6684cf91023287e0a952af9
- system prompt (both arms): 'You are a helpful Python programmer. Write the requested function.'
- decoding: {'num_ctx': 4096, 'num_predict': 2048, 'repeat_penalty': 1.0, 'temperature': 0.0}, one greedy reply per problem
- `think` flag sent: base False, ft5 False (None: omitted); capabilities reported: base ['tools', 'thinking', 'completion'], ft5 ['tools', 'thinking', 'completion']
- Ollama 0.40.4

## Result

| arm | passed | pass@1 % | Wilson 95 % CI | binomial SE |
|---|---|---|---|---|
| base | 218/300 | 72.7 | [67.4, 77.4] | 2.6 |
| ft5 | 210/300 | 70.0 | [64.6, 74.9] | 2.6 |

Paired on the same 300 problems: both pass 193, only base 25, only ft5 17, neither 65.

- **drop = pass@1(base) - pass@1(ft5) = +2.7 points** (positive: ft5 is worse)
- 95 % CI of the drop (Newcombe method 10): [-1.6, +6.9] points; Wald paired SE 2.2
- exact McNemar p = 0.28 (42 discordant problems)
- criterion 3 on the point estimate (drop <= 3 points): **MET**
- non-inferiority shown (upper end of the drop's 95 % CI <= 3 points): **no** (the point estimate passes but the interval is too wide to exclude a larger drop)

## What the failures were

| class | base | ft5 |
|---|---|---|
| pass | 218 | 210 |
| wrong | 55 | 70 |
| runtime | 17 | 17 |
| syntax | 10 | 0 |
| timeout | 0 | 3 |

| | base | ft5 |
|---|---|---|
| replies cut at num_predict (graded as they are) | 12 | 0 |
| reply format (first-block extraction path) | {'fence': 288, 'fence-unclosed': 12} | {'fence': 300} |
| code taken from a later block (first block lacked the entry point) | 0 | 0 |
| blocked by the safety screen (counted as failures) | 0  | 0  |
| answers the project lint would flag (info only: lint is not enforced here) | 11 {'syntax': 10, 'import': 1} | 1 {'import': 1} |
| passes if the project lint were enforced | 217 | 209 |
| grader errors | 0 | 0 |

## Time per problem

| | base | ft5 |
|---|---|---|
| generation wall, mean s | 5.9 | 3.3 |
| generation wall, median s | 4.1 | 3.0 |
| generation wall, p90 s | 8.2 | 4.7 |
| tokens per answer, mean  | 304 | 147 |
| decode tok/s, median  | 53.5 | 48.3 |

Generation total: base 29.7 min, ft5 16.3 min. Sandbox time per executed answer: mean 0.64 s, p90 0.09 s, max 90.09 s.

Check that the neutral system prompt replaced each model's own SYSTEM: warm-up prompt tokens base 67, ft5 67 (equal: yes)

## Problems where the arms differ (42)

| problem | passed | the other arm failed as |
|---|---|---|
| Mbpp/125 | base | wrong AssertionError |
| Mbpp/126 | base | runtime TypeError |
| Mbpp/129 | ft5 | wrong AssertionError |
| Mbpp/135 | ft5 | runtime ValueError |
| Mbpp/142 | ft5 | runtime ValueError |
| Mbpp/161 | ft5 | runtime TypeError |
| Mbpp/166 | base | wrong AssertionError |
| Mbpp/20 | base | timeout  |
| Mbpp/245 | base | wrong AssertionError |
| Mbpp/251 | base | wrong AssertionError |
| Mbpp/259 | base | runtime ValueError |
| Mbpp/264 | ft5 | syntax SyntaxError |
| Mbpp/265 | base | wrong AssertionError |
| Mbpp/290 | base | wrong AssertionError |
| Mbpp/293 | ft5 | wrong AssertionError |
| Mbpp/301 | base | runtime AttributeError |
| Mbpp/392 | ft5 | wrong AssertionError |
| Mbpp/426 | base | wrong AssertionError |
| Mbpp/432 | base | wrong AssertionError |
| Mbpp/435 | base | wrong AssertionError |
| Mbpp/437 | base | wrong AssertionError |
| Mbpp/446 | base | wrong AssertionError |
| Mbpp/473 | base | wrong AssertionError |
| Mbpp/475 | base | wrong AssertionError |
| Mbpp/564 | ft5 | wrong AssertionError |
| Mbpp/579 | base | runtime TypeError |
| Mbpp/600 | ft5 | wrong AssertionError |
| Mbpp/63 | ft5 | wrong AssertionError |
| Mbpp/631 | base | wrong AssertionError |
| Mbpp/641 | ft5 | runtime ValueError |
| Mbpp/643 | ft5 | wrong AssertionError |
| Mbpp/721 | base | wrong AssertionError |
| Mbpp/734 | ft5 | syntax SyntaxError |
| Mbpp/741 | base | wrong AssertionError |
| Mbpp/745 | base | wrong AssertionError |
| Mbpp/763 | base | runtime AttributeError |
| Mbpp/772 | ft5 | wrong AssertionError |
| Mbpp/781 | base | timeout  |
| Mbpp/79 | base | wrong AssertionError |
| Mbpp/80 | ft5 | runtime ValueError |
| Mbpp/84 | ft5 | wrong AssertionError |
| Mbpp/9 | ft5 | syntax SyntaxError |
