# Forgetting check: ft vs base on MBPP+ (300 problems)

Criterion 3 (LOGBOOK, frozen 2026-10-22): the fine-tuned model may lose at most 3 points of pass@1 against the same base.

- reference arm A = `base` (base-qwen3.5-9b), candidate arm B = `ft` (opencv-expert-q4)
- dataset evalplus/mbppplus @ b2d74c91837c3f2a20c1299ae98133cbe7cfa077; slice file sha256 4820ab7de00c34df1f419ac26c73ba97beff8984c6684cf91023287e0a952af9
- system prompt (both arms): 'You are a helpful Python programmer. Write the requested function.'
- decoding: {'num_ctx': 4096, 'num_predict': 2048, 'repeat_penalty': 1.0, 'temperature': 0.0}, one greedy reply per problem
- `think` flag sent: base False, ft False (None: omitted); capabilities reported: base ['tools', 'thinking', 'completion'], ft ['tools', 'thinking', 'completion']
- Ollama 0.40.4

## Result

| arm | passed | pass@1 % | Wilson 95 % CI | binomial SE |
|---|---|---|---|---|
| base | 218/300 | 72.7 | [67.4, 77.4] | 2.6 |
| ft | 216/300 | 72.0 | [66.7, 76.8] | 2.6 |

Paired on the same 300 problems: both pass 197, only base 21, only ft 19, neither 63.

- **drop = pass@1(base) - pass@1(ft) = +0.7 points** (positive: ft is worse)
- 95 % CI of the drop (Newcombe method 10): [-3.5, +4.9] points; Wald paired SE 2.1
- exact McNemar p = 0.875 (40 discordant problems)
- criterion 3 on the point estimate (drop <= 3 points): **MET**
- non-inferiority shown (upper end of the drop's 95 % CI <= 3 points): **no** (the point estimate passes but the interval is too wide to exclude a larger drop)

## What the failures were

| class | base | ft |
|---|---|---|
| pass | 218 | 216 |
| wrong | 55 | 67 |
| runtime | 17 | 14 |
| syntax | 10 | 0 |
| timeout | 0 | 3 |

| | base | ft |
|---|---|---|
| replies cut at num_predict (graded as they are) | 12 | 0 |
| reply format (first-block extraction path) | {'fence': 288, 'fence-unclosed': 12} | {'fence': 300} |
| code taken from a later block (first block lacked the entry point) | 0 | 0 |
| blocked by the safety screen (counted as failures) | 0  | 0  |
| answers the project lint would flag (info only: lint is not enforced here) | 11 {'syntax': 10, 'import': 1} | 1 {'import': 1} |
| passes if the project lint were enforced | 217 | 215 |
| grader errors | 0 | 0 |

## Time per problem

| | base | ft |
|---|---|---|
| generation wall, mean s | 5.9 | 2.7 |
| generation wall, median s | 4.1 | 2.7 |
| generation wall, p90 s | 8.2 | 3.9 |
| tokens per answer, mean  | 304 | 135 |
| decode tok/s, median  | 53.5 | 53.7 |

Generation total: base 29.7 min, ft 13.7 min. Sandbox time per executed answer: mean 0.55 s, p90 0.09 s, max 90.09 s.

Check that the neutral system prompt replaced each model's own SYSTEM: warm-up prompt tokens base 67, ft 67 (equal: yes)

## Problems where the arms differ (40)

| problem | passed | the other arm failed as |
|---|---|---|
| Mbpp/120 | base | wrong AssertionError |
| Mbpp/126 | base | timeout  |
| Mbpp/135 | ft | runtime ValueError |
| Mbpp/16 | base | wrong AssertionError |
| Mbpp/161 | ft | runtime TypeError |
| Mbpp/165 | base | wrong AssertionError |
| Mbpp/20 | base | wrong AssertionError |
| Mbpp/251 | base | wrong AssertionError |
| Mbpp/259 | base | wrong AssertionError |
| Mbpp/293 | ft | wrong AssertionError |
| Mbpp/392 | ft | wrong AssertionError |
| Mbpp/429 | base | wrong AssertionError |
| Mbpp/432 | base | wrong AssertionError |
| Mbpp/435 | base | wrong AssertionError |
| Mbpp/437 | base | wrong AssertionError |
| Mbpp/446 | base | wrong AssertionError |
| Mbpp/473 | base | wrong AssertionError |
| Mbpp/475 | base | wrong AssertionError |
| Mbpp/564 | ft | wrong AssertionError |
| Mbpp/600 | ft | wrong AssertionError |
| Mbpp/631 | base | wrong AssertionError |
| Mbpp/633 | ft | wrong AssertionError |
| Mbpp/641 | ft | runtime ValueError |
| Mbpp/643 | ft | wrong AssertionError |
| Mbpp/67 | ft | runtime IndexError |
| Mbpp/734 | ft | syntax SyntaxError |
| Mbpp/737 | base | wrong AssertionError |
| Mbpp/741 | base | wrong AssertionError |
| Mbpp/742 | base | runtime TypeError |
| Mbpp/745 | base | wrong AssertionError |
| Mbpp/763 | base | runtime AttributeError |
| Mbpp/772 | ft | wrong AssertionError |
| Mbpp/773 | ft | wrong AssertionError |
| Mbpp/781 | base | timeout  |
| Mbpp/785 | ft | runtime ValueError |
| Mbpp/80 | ft | runtime ValueError |
| Mbpp/800 | ft | wrong AssertionError |
| Mbpp/801 | ft | wrong AssertionError |
| Mbpp/84 | ft | wrong AssertionError |
| Mbpp/9 | ft | syntax SyntaxError |
