# Forgetting check: ft5 vs ft on MBPP+ (300 problems)

Criterion 3 (LOGBOOK, frozen 2026-10-22): the fine-tuned model may lose at most 3 points of pass@1 against the same base.

- reference arm A = `ft` (opencv-expert-q4), candidate arm B = `ft5` (opencv-expert-q5)
- dataset evalplus/mbppplus @ b2d74c91837c3f2a20c1299ae98133cbe7cfa077; slice file sha256 4820ab7de00c34df1f419ac26c73ba97beff8984c6684cf91023287e0a952af9
- system prompt (both arms): 'You are a helpful Python programmer. Write the requested function.'
- decoding: {'num_ctx': 4096, 'num_predict': 2048, 'repeat_penalty': 1.0, 'temperature': 0.0}, one greedy reply per problem
- `think` flag sent: ft False, ft5 False (None: omitted); capabilities reported: ft ['tools', 'thinking', 'completion'], ft5 ['tools', 'thinking', 'completion']
- Ollama 0.40.4

## Result

| arm | passed | pass@1 % | Wilson 95 % CI | binomial SE |
|---|---|---|---|---|
| ft | 216/300 | 72.0 | [66.7, 76.8] | 2.6 |
| ft5 | 210/300 | 70.0 | [64.6, 74.9] | 2.6 |

Paired on the same 300 problems: both pass 200, only ft 16, only ft5 10, neither 74.

- **drop = pass@1(ft) - pass@1(ft5) = +2.0 points** (positive: ft5 is worse)
- 95 % CI of the drop (Newcombe method 10): [-1.4, +5.4] points; Wald paired SE 1.7
- exact McNemar p = 0.327 (26 discordant problems)
- criterion 3 on the point estimate (drop <= 3 points): **MET**
- non-inferiority shown (upper end of the drop's 95 % CI <= 3 points): **no** (the point estimate passes but the interval is too wide to exclude a larger drop)

## What the failures were

| class | ft | ft5 |
|---|---|---|
| pass | 216 | 210 |
| wrong | 67 | 70 |
| runtime | 14 | 17 |
| timeout | 3 | 3 |

| | ft | ft5 |
|---|---|---|
| replies cut at num_predict (graded as they are) | 0 | 0 |
| reply format (first-block extraction path) | {'fence': 300} | {'fence': 300} |
| code taken from a later block (first block lacked the entry point) | 0 | 0 |
| blocked by the safety screen (counted as failures) | 0  | 0  |
| answers the project lint would flag (info only: lint is not enforced here) | 1 {'import': 1} | 1 {'import': 1} |
| passes if the project lint were enforced | 215 | 209 |
| grader errors | 0 | 0 |

## Time per problem

| | ft | ft5 |
|---|---|---|
| generation wall, mean s | 2.7 | 3.3 |
| generation wall, median s | 2.7 | 3.0 |
| generation wall, p90 s | 3.9 | 4.7 |
| tokens per answer, mean  | 135 | 147 |
| decode tok/s, median  | 53.7 | 48.3 |

Generation total: ft 13.7 min, ft5 16.3 min. Sandbox time per executed answer: mean 1.12 s, p90 0.09 s, max 90.09 s.

Check that the neutral system prompt replaced each model's own SYSTEM: warm-up prompt tokens ft 67, ft5 67 (equal: yes)

## Problems where the arms differ (26)

| problem | passed | the other arm failed as |
|---|---|---|
| Mbpp/120 | ft5 | wrong AssertionError |
| Mbpp/125 | ft | wrong AssertionError |
| Mbpp/129 | ft5 | wrong AssertionError |
| Mbpp/142 | ft5 | runtime IndexError |
| Mbpp/16 | ft5 | wrong AssertionError |
| Mbpp/165 | ft5 | wrong AssertionError |
| Mbpp/166 | ft | wrong AssertionError |
| Mbpp/245 | ft | wrong AssertionError |
| Mbpp/264 | ft5 | wrong AssertionError |
| Mbpp/265 | ft | wrong AssertionError |
| Mbpp/290 | ft | wrong AssertionError |
| Mbpp/301 | ft | runtime AttributeError |
| Mbpp/426 | ft | wrong AssertionError |
| Mbpp/429 | ft5 | wrong AssertionError |
| Mbpp/579 | ft | runtime TypeError |
| Mbpp/63 | ft5 | wrong AssertionError |
| Mbpp/633 | ft | wrong AssertionError |
| Mbpp/67 | ft | wrong AssertionError |
| Mbpp/721 | ft | wrong AssertionError |
| Mbpp/737 | ft5 | wrong AssertionError |
| Mbpp/742 | ft5 | runtime TypeError |
| Mbpp/773 | ft | wrong AssertionError |
| Mbpp/785 | ft | runtime ValueError |
| Mbpp/79 | ft | wrong AssertionError |
| Mbpp/800 | ft | wrong AssertionError |
| Mbpp/801 | ft | wrong AssertionError |
