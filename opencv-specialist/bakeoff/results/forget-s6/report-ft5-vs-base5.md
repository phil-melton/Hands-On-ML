# Forgetting check: ft5 vs base5 on MBPP+ (300 problems)

Criterion 3 (LOGBOOK, frozen 2026-10-22): the fine-tuned model may lose at most 3 points of pass@1 against the same base.

- reference arm A = `base5` (base-qwen3.5-9b-q5), candidate arm B = `ft5` (opencv-expert-q5)
- dataset evalplus/mbppplus @ b2d74c91837c3f2a20c1299ae98133cbe7cfa077; slice file sha256 4820ab7de00c34df1f419ac26c73ba97beff8984c6684cf91023287e0a952af9
- system prompt (both arms): 'You are a helpful Python programmer. Write the requested function.'
- decoding: {'num_ctx': 4096, 'num_predict': 2048, 'repeat_penalty': 1.0, 'temperature': 0.0}, one greedy reply per problem
- `think` flag sent: base5 False, ft5 False (None: omitted); capabilities reported: base5 ['tools', 'thinking', 'completion'], ft5 ['tools', 'thinking', 'completion']
- Ollama 0.40.4

## Result

| arm | passed | pass@1 % | Wilson 95 % CI | binomial SE |
|---|---|---|---|---|
| base5 | 222/300 | 74.0 | [68.8, 78.6] | 2.5 |
| ft5 | 210/300 | 70.0 | [64.6, 74.9] | 2.6 |

Paired on the same 300 problems: both pass 194, only base5 28, only ft5 16, neither 62.

- **drop = pass@1(base5) - pass@1(ft5) = +4.0 points** (positive: ft5 is worse)
- 95 % CI of the drop (Newcombe method 10): [-0.4, +8.4] points; Wald paired SE 2.2
- exact McNemar p = 0.0961 (44 discordant problems)
- criterion 3 on the point estimate (drop <= 3 points): **NOT MET**
- non-inferiority shown (upper end of the drop's 95 % CI <= 3 points): **no**

## What the failures were

| class | base5 | ft5 |
|---|---|---|
| pass | 222 | 210 |
| wrong | 52 | 70 |
| runtime | 20 | 17 |
| syntax | 6 | 0 |
| timeout | 0 | 3 |

| | base5 | ft5 |
|---|---|---|
| replies cut at num_predict (graded as they are) | 7 | 0 |
| reply format (first-block extraction path) | {'fence': 294, 'bare': 4, 'fence-unclosed': 2} | {'fence': 300} |
| code taken from a later block (first block lacked the entry point) | 0 | 0 |
| blocked by the safety screen (counted as failures) | 0  | 0  |
| answers the project lint would flag (info only: lint is not enforced here) | 7 {'syntax': 6, 'import': 1} | 1 {'import': 1} |
| passes if the project lint were enforced | 221 | 209 |
| grader errors | 0 | 0 |

## Time per problem

| | base5 | ft5 |
|---|---|---|
| generation wall, mean s | 6.5 | 3.3 |
| generation wall, median s | 4.3 | 3.0 |
| generation wall, p90 s | 11.3 | 4.7 |
| tokens per answer, mean  | 301 | 147 |
| decode tok/s, median  | 48.1 | 48.3 |

Generation total: base5 32.5 min, ft5 16.3 min. Sandbox time per executed answer: mean 0.65 s, p90 0.09 s, max 90.09 s.

Check that the neutral system prompt replaced each model's own SYSTEM: warm-up prompt tokens base5 67, ft5 67 (equal: yes)

## Problems where the arms differ (44)

| problem | passed | the other arm failed as |
|---|---|---|
| Mbpp/100 | ft5 | wrong AssertionError |
| Mbpp/11 | ft5 | wrong AssertionError |
| Mbpp/123 | ft5 | wrong AssertionError |
| Mbpp/125 | base5 | wrong AssertionError |
| Mbpp/126 | base5 | runtime TypeError |
| Mbpp/129 | ft5 | wrong AssertionError |
| Mbpp/130 | ft5 | wrong AssertionError |
| Mbpp/145 | ft5 | runtime ValueError |
| Mbpp/161 | ft5 | runtime TypeError |
| Mbpp/166 | base5 | wrong AssertionError |
| Mbpp/20 | base5 | timeout  |
| Mbpp/245 | base5 | wrong AssertionError |
| Mbpp/251 | base5 | wrong AssertionError |
| Mbpp/264 | ft5 | wrong AssertionError |
| Mbpp/265 | base5 | wrong AssertionError |
| Mbpp/279 | ft5 | syntax SyntaxError |
| Mbpp/290 | base5 | wrong AssertionError |
| Mbpp/292 | ft5 | wrong AssertionError |
| Mbpp/426 | base5 | wrong AssertionError |
| Mbpp/432 | base5 | wrong AssertionError |
| Mbpp/435 | base5 | wrong AssertionError |
| Mbpp/437 | base5 | wrong AssertionError |
| Mbpp/459 | base5 | wrong AssertionError |
| Mbpp/473 | base5 | wrong AssertionError |
| Mbpp/475 | base5 | wrong AssertionError |
| Mbpp/556 | ft5 | runtime ValueError |
| Mbpp/564 | ft5 | wrong AssertionError |
| Mbpp/572 | base5 | wrong AssertionError |
| Mbpp/579 | base5 | runtime TypeError |
| Mbpp/581 | base5 | wrong AssertionError |
| Mbpp/631 | base5 | wrong AssertionError |
| Mbpp/641 | ft5 | wrong AssertionError |
| Mbpp/69 | ft5 | wrong AssertionError |
| Mbpp/721 | base5 | wrong AssertionError |
| Mbpp/741 | base5 | wrong AssertionError |
| Mbpp/745 | base5 | wrong AssertionError |
| Mbpp/753 | ft5 | runtime ValueError |
| Mbpp/763 | base5 | runtime AttributeError |
| Mbpp/773 | base5 | wrong AssertionError |
| Mbpp/781 | base5 | timeout  |
| Mbpp/79 | base5 | wrong AssertionError |
| Mbpp/80 | ft5 | runtime ValueError |
| Mbpp/801 | base5 | wrong AssertionError |
| Mbpp/809 | base5 | wrong AssertionError |
