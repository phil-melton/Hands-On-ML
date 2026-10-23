# Forgetting check: base5 vs base on MBPP+ (300 problems)

Criterion 3 (LOGBOOK, frozen 2026-10-22): the fine-tuned model may lose at most 3 points of pass@1 against the same base.

- reference arm A = `base` (base-qwen3.5-9b), candidate arm B = `base5` (base-qwen3.5-9b-q5)
- dataset evalplus/mbppplus @ b2d74c91837c3f2a20c1299ae98133cbe7cfa077; slice file sha256 4820ab7de00c34df1f419ac26c73ba97beff8984c6684cf91023287e0a952af9
- system prompt (both arms): 'You are a helpful Python programmer. Write the requested function.'
- decoding: {'num_ctx': 4096, 'num_predict': 2048, 'repeat_penalty': 1.0, 'temperature': 0.0}, one greedy reply per problem
- `think` flag sent: base False, base5 False (None: omitted); capabilities reported: base ['tools', 'thinking', 'completion'], base5 ['tools', 'thinking', 'completion']
- Ollama 0.40.4

## Result

| arm | passed | pass@1 % | Wilson 95 % CI | binomial SE |
|---|---|---|---|---|
| base | 218/300 | 72.7 | [67.4, 77.4] | 2.6 |
| base5 | 222/300 | 74.0 | [68.8, 78.6] | 2.5 |

Paired on the same 300 problems: both pass 205, only base 13, only base5 17, neither 65.

- **drop = pass@1(base) - pass@1(base5) = -1.3 points** (positive: base5 is worse)
- 95 % CI of the drop (Newcombe method 10): [-5.0, +2.3] points; Wald paired SE 1.8
- exact McNemar p = 0.585 (30 discordant problems)
- criterion 3 on the point estimate (drop <= 3 points): **MET**
- non-inferiority shown (upper end of the drop's 95 % CI <= 3 points): **yes**

## What the failures were

| class | base | base5 |
|---|---|---|
| pass | 218 | 222 |
| wrong | 55 | 52 |
| runtime | 17 | 20 |
| syntax | 10 | 6 |

| | base | base5 |
|---|---|---|
| replies cut at num_predict (graded as they are) | 12 | 7 |
| reply format (first-block extraction path) | {'fence': 288, 'fence-unclosed': 12} | {'fence': 294, 'bare': 4, 'fence-unclosed': 2} |
| code taken from a later block (first block lacked the entry point) | 0 | 0 |
| blocked by the safety screen (counted as failures) | 0  | 0  |
| answers the project lint would flag (info only: lint is not enforced here) | 11 {'syntax': 10, 'import': 1} | 7 {'syntax': 6, 'import': 1} |
| passes if the project lint were enforced | 217 | 221 |
| grader errors | 0 | 0 |

## Time per problem

| | base | base5 |
|---|---|---|
| generation wall, mean s | 5.9 | 6.5 |
| generation wall, median s | 4.1 | 4.3 |
| generation wall, p90 s | 8.2 | 11.3 |
| tokens per answer, mean  | 304 | 301 |
| decode tok/s, median  | 53.5 | 48.1 |

Generation total: base 29.7 min, base5 32.5 min. Sandbox time per executed answer: mean 0.10 s, p90 0.09 s, max 2.30 s.

Check that the neutral system prompt replaced each model's own SYSTEM: warm-up prompt tokens base 67, base5 67 (equal: yes)

## Problems where the arms differ (30)

| problem | passed | the other arm failed as |
|---|---|---|
| Mbpp/100 | base | wrong AssertionError |
| Mbpp/11 | base | wrong AssertionError |
| Mbpp/123 | base | wrong AssertionError |
| Mbpp/130 | base | wrong AssertionError |
| Mbpp/135 | base5 | runtime ValueError |
| Mbpp/142 | base5 | runtime ValueError |
| Mbpp/145 | base | runtime ValueError |
| Mbpp/259 | base | wrong AssertionError |
| Mbpp/279 | base | syntax SyntaxError |
| Mbpp/292 | base | wrong AssertionError |
| Mbpp/293 | base5 | wrong AssertionError |
| Mbpp/301 | base | runtime TypeError |
| Mbpp/392 | base5 | wrong AssertionError |
| Mbpp/446 | base | wrong AssertionError |
| Mbpp/459 | base5 | wrong AssertionError |
| Mbpp/556 | base | runtime ValueError |
| Mbpp/572 | base5 | wrong AssertionError |
| Mbpp/581 | base5 | wrong AssertionError |
| Mbpp/600 | base5 | wrong AssertionError |
| Mbpp/63 | base5 | wrong AssertionError |
| Mbpp/643 | base5 | wrong AssertionError |
| Mbpp/69 | base | wrong AssertionError |
| Mbpp/734 | base5 | syntax SyntaxError |
| Mbpp/753 | base | runtime ValueError |
| Mbpp/772 | base5 | wrong AssertionError |
| Mbpp/773 | base5 | wrong AssertionError |
| Mbpp/801 | base5 | wrong AssertionError |
| Mbpp/809 | base5 | wrong AssertionError |
| Mbpp/84 | base5 | wrong AssertionError |
| Mbpp/9 | base5 | syntax SyntaxError |
