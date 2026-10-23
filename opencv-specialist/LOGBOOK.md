# opencv-expert logbook

**Next step:** Stage 7 (MCP server `serve/server.py` with runtime metamorphic checks, `claude mcp add`, 5-task smoke
test, the 20-task E2E set for criterion 4), then the final report (⛔). Stage 6 ships `opencv-expert` = Q5_K_M
(`serve/Modelfile`); its test-set and MBPP+ results are below, with the like-for-like Q5 base comparator `r6-9b-notes-q5`.
Colab spend so far 39.44 CU (Stage 4 23.15, Stage 5 16.29); Stages 6-7 run locally (0 CU).

## 2026-10-23 Stage 7 built: MCP server, runtime metamorphic checks, E2E set; prediction

Built by two Sonnet subagents, re-checked by me. `serve/server.py` (mcp 2.3.0, MIT; venv `~/venvs/serve`,
`serve/requirements-serve.txt`): one tool, `opencv_solve(task, images, outputs, params, max_repairs=3, metamorphic=True)`.
The user message is `TaskSpec.to_prompt()`; the request is the Stage 6 evaluation's (SYSTEM, `think: false`, greedy
options, no JSON-schema decoding); code via the bake-off's extractor; `sandbox.run_solution`; repairs in the training
repair-pair shape (system, user, assistant ```` ```python ```` block, user "That failed: ...", 4 messages; only the latest
failure is replayed). Earlier assistant turns render plain in both the training template and the Ollama TEMPLATE.
`serve/metamorphic.py`: reruns verified code on rot90 / mirror / dim (x0.6) copies and classifies each output as
invariant / covariant / changed / error (tolerances from pixel-grid arguments: 1 % permute, 5 % dim, 2 px coordinates);
a confidence signal, not a gate. Registered with `claude mcp add opencv-expert --scope local` (this project only; `claude mcp
get` -> Connected). Live: 2 calls, both verified at attempt 1 (51.7 s incl. the model load, then 15.7 s), metamorphic
stable (counts and areas invariant, segments covariant).
`serve/e2e/`: 20 multi-step tasks (3-4 steps, all 9 families, 6 with held-out families; seeds 6.01M-6.21M, disjoint
from the splits), images and exact truths manufactured from the family generators; tolerances imported from the
family verifiers or propagated by worst-case bounds (wide: task 2 centroid ~19 px, task 19 angle ~7 deg); every task's
reference pipeline (the families' REFERENCE solutions through the sandbox) passes; perturbed answers fail. Runner: one
`claude.exe -p` session per task and arm, the same prompt in both arms; both get Bash / Read / Write (Python via
`wsl.exe ... ~/venvs/cv5/bin/python`); the specialist arm adds the MCP server through `--mcp-config`;
`--strict-mcp-config --setting-sources '' --disable-slash-commands`, auto-memory and CLAUDE.md off, `dontAsk`.
Tests: 46 (serve) + 98 (e2e); full suite 1,270 passed, 4 skipped.
Orchestrator: Sonnet 5.5 (`--model sonnet`), signed in through Phil's subscription (no API key: plan usage, $0 API).
Prediction (Rule 1): alone 20 x ~400k tokens (the `total` counter, mostly cache reads) and ~3 min per task; specialist
20 x ~250k and ~2.5 min: ~13M tokens, ~1.8 h sequential. First one task per arm to check the plumbing.

## 2026-10-23 Stage 6 evaluation: test set, quant ladder, forgetting; ship Q5_K_M (local GPU, 0 CU)

Ran (predicted above): `s6-q4/q5/q8` in run `bakeoff-v2-20261022` (238 test tasks, 112 distinct prompts, greedy,
num_predict 2048, num_ctx 8192, same system prompt as r5 but no notes), paired with r5 (base + notes, Q4_K_M); MBPP+
for the fine-tuned Q4 and Q5 against the base. Measured generation: Q4 23.3 min, Q5 22.7, Q8 30.6 (predicted 26 / 30 /
41); MBPP+ base 29.7 min, Q4 13.7, Q5 16.3 (predicted 40-95 / 15-40). The watchdog fired once (02:31, commit 2.99 GB): it
killed the last Q8 chunk, whose 3 missing replies were re-run. 0 grader errors. Results in `bakeoff/results/`.

pass@1 on the test set, % (binomial SE ~3; SE by prompt 4-5 for `all`):

| group | n / prompts | r5 base+notes Q4 | ft Q4_K_M | ft Q5_K_M | ft Q8_0 | r6 base+notes Q5 |
|---|---|---|---|---|---|---|
| all | 238 / 112 | 38.7 | 61.3 | **76.9** | 71.4 | 35.3 |
| 5.0 subset | 149 / 65 | 20.1 | 48.3 | **71.1** | 63.8 | 13.4 |
| families, 5.0 | 132 / 48 | 16.7 | 48.5 | 74.2 | 64.4 | 9.8 |
| families, timeless | 66 / 24 | 68.2 | 90.9 | 92.4 | 92.4 | 71.2 |
| trained families | 154 / 56 | 31.2 | 70.1 | 92.2 | 83.8 | 26.0 |
| held-out families | 44 / 16 | 43.2 | 36.4 | 38.6 | 38.6 | 45.5 |
| gotchas | 40 / 40 | 62.5 | 55.0 | 60.0 | 60.0 | 60.0 |

The like-for-like comparator `r6-9b-notes-q5` (quantize 246 s, test set 36.0 min; predicted ~5 and ~40) scores no
better than r5: 35.3 vs 38.7 % (p by prompt 0.71). Its per-family scores swing by up to 35 pts between the two quants
(contour 81.8 -> 50.0, perspective 36.4 -> 72.7, homography 36.4 -> 9.1, components 45.5 -> 13.6), so the base's
per-family numbers on 8 prompts are mostly greedy-trajectory noise; the fine-tune's sit at the ceiling in 6 of 7
trained families. So the dev gap base Ollama-Q4 vs HF-bf16 (-11.2) is not specific to Q4_K_M; whether it is
quantization in general or the llama.cpp path is not measured (no base Q8 or bf16 rung on the test set).
Per family (Q5): contour 100, hough 100 (r5 4.5), perspective 100 (36.4), homography 100 (36.4), histogram 100 (13.6),
components 100 (45.5), text_layout 45.5 (0.0); held out: chessboard 0 (0), color_segmentation 77.3 (86.4).

**Quant ladder, paired (exact sign test over prompts):** Q4 - Q5 = -15.5 pts (9 vs 46 discordant tasks, p 0.006); Q5 - Q8
= +5.5 (p 0.27); Q4 - Q8 = -10.1 (p 0.076). **Decision: ship Q5_K_M** (6.19 GiB, 47.8 tok/s, fits the 12 GB GPU with
room): AGENT_PROMPT's rule is the smallest quant within noise of the best, and Q4 is not. `serve/Modelfile`, Ollama tag
`opencv-expert` (same blob as `opencv-expert-q5`, sha256 e1626a67...). `train.config.OLLAMA_PARAMETERS` now also sets
repeat_penalty 1.0 and num_predict 2048, the values every evaluation request sent, so a client sending no options gets
the evaluated behaviour. Deviation from the frozen wording ("measured on the shipped Q4_K_M"): the criteria below are
reported for both Q4 (as frozen) and Q5 (as shipped), and Q5 also against the Q5 base (like-for-like quantization).

**Audit of Q4 vs Q5.** Q4's extra failures are not 5.0-API errors (class `api`: Q4 12, Q5 15, Q8 7). They are ordinary
bugs in the families the fine-tune changed most: hough 50 vs 100 % (8 tasks unpack `cv2.minAreaRect`'s 3-tuple into two
names, a timeless bug), components 50 vs 100 % (9 tasks build stats by hand from `connectedComponents` instead of
calling `connectedComponentsWithStats`), text_layout 22.7 vs 45.5, homography 77.3 vs 100. Dev validation had shown only
-4.0 (36 prompts, underpowered). Q8 below Q5 (-5.5, n.s.) matches dev too; its extra failures are hough `algorithm` (9)
and text_layout; unexplained, within noise.
**PLAN §7 Q5 evidence (Rule 10; the answer is Phil's).** Prediction: "the 5.0-specific behaviours break first".
Q4 - Q5: families/5.0 -25.7 pts, families/timeless -1.5. But the api-error count did not rise at Q4, and
perspective_rectify (timeless, +63.6 from fine-tuning) kept 100 % at Q4 while hough (5.0, +95.5) fell to 50 %.
bf16 on the test set was not run (needs a gen-only Colab path, ~2 CU); on dev, bf16 82.8 vs Q5 79.8 (-3.0).

MBPP+ forgetting (300 problems, neutral system prompt, greedy): base Q4 72.7 %; ft Q4 72.0 (drop +0.7, 95 % CI
[-3.5, +4.9], McNemar p 0.875); ft Q5 70.0 (drop +2.7, CI [-1.6, +6.9], p 0.28: 25 lost, 17 gained; `wrong` 55 -> 70,
syntax 10 -> 0 because the base runs past num_predict 12 times); base Q5 74.0 (32.5 min; predicted ~33), so like-for-like the
ft Q5 drop is +4.0 (CI [-0.4, +8.4], McNemar p 0.096; 28 lost, 16 gained). Quantization alone moves MBPP+ by 1-2 pts
either way (base Q5 - Q4 +1.3, p 0.59; ft Q5 - Q4 -2.0, p 0.33). The watchdog fired once more (04:01, commit 2.83 GB)
because I ran the full test suite on the CPU during generation; it cost the reply in flight. Never again during Ollama runs.

**Success criteria (frozen 2026-10-22), target vs measured:**

| criterion | target | ft Q4 vs r5 (as frozen) | ft Q5 vs r5 | ft Q5 vs r6 (like-for-like) |
|---|---|---|---|---|
| 1a specialization, test | >= +10, p < 0.05 | +22.7, p 0.006: met | +38.2, p 1e-7: met | +41.6, p 8e-9: met |
| 1b specialization, 5.0 subset | >= +25, p < 0.05 | +28.2, p 0.003: met | +51.0, p 3e-8: met | +57.7, p 6e-11: met |
| 2 generalization | held-out gain >= half in-family | -6.8 vs +19.5 needed: **not met** | -4.5 (CI [-13.5, +4.5]) vs +30.5: **not met** | -6.8 (CI [-20.1, +6.5]) vs +33.1: **not met** |
| 3 forgetting (MBPP+) | drop <= 3 | +0.7: met (CI to +4.9) | +2.7: met (CI to +6.9) | +4.0 (CI [-0.4, +8.4], p 0.096): **not met** |
| 4 integration | Stage 7 | - | - | - |

Generalization fails outright, not for lack of power: the upper CI end is below the target. Chessboard stays 0 % for
every model (r5 too), and color_segmentation loses 2 prompts (86.4 -> 77.3). The fine-tune learned the trained
families, not a transferable "OpenCV 5.0 skill". Forgetting passes on the point estimate in every comparison, but no
comparison shows non-inferiority (every CI's upper end exceeds 3 points),
and like-for-like the shipped Q5 misses the point threshold too (+4.0, not significant). Q4 would pass criterion 3
like-for-like (+0.7) but loses 15.5 pts on the test set (p 0.006); the two fine-tunes differ by 2.0 pts on MBPP+ (p 0.33).
I keep Q5 and flag it in the final report.

## 2026-10-23 Stage 6: merge, GGUF ladder, validation (local, 0 CU)

Built by two Sonnet subagents. `train/merge_lora.py`: streaming merge of r100 @ e1 into the bf16 base, numpy only,
1.5 GiB peak RSS, 335 s; all 248 LoRA pairs (24 linear-attention x 5, 32 MLP x 3, 8 full-attention x 4) used once,
scale 1.0; the 527 untouched tensors and the headers byte-identical; every merged element within half a bf16 ulp plus
a float32 allowance. GGUF ladder through the Stage 3 converter: Q8_0 9.11 GiB, Q5_K_M 6.19, Q4_K_M 5.38 GiB (tensor-type
mix equal to the base's); Ollama `opencv-expert-q4/q5/q8`; Q8 fits the 12 GB GPU fully (8.26 GiB at num_ctx 8192).
Explicit system message replaces the Modelfile SYSTEM (prompt tokens 85 either way, = `render_turn`); `think: false`
gives no thinking. GGUF metadata lacks the license / base-model keys (no README in the merged folder; cosmetic).
Dev validation (36 prompts / 99 tasks, greedy): Q4 75.8, Q5 79.8, Q8 66.7 % vs Colab HF bf16 82.8; Q4 vs Colab -7.0
(sign test p 0.25), Q8 -16.2 (p 0.031). Control: the untrained base scores 34.3 % through Ollama Q4 vs 45.5 % through
HF bf16 (-11.2 with no LoRA), so the serving path accounts for most of the gap; the fine-tune gain survives (+41.5
through Ollama vs +37.3 on Colab). Q8 lowest is unexplained (36 prompts, SE ~8). MBPP+ forgetting slice:
`evalplus/mbppplus` @ b2d74c91 (Apache-2.0; MBPP CC-BY-4.0), 300 problems, 3 dropped for 13-gram overlap with the
training / replay solutions; harness `bakeoff/forgetting.py` (neutral system prompt, same for both arms; EvalPlus
tests run in the sandbox). Suite 1,172 passed, 3 skipped.
**Memory finding:** while an Ollama model serves sustained generation, free Windows commit falls ~0.6-0.85 GB per
minute until the model unloads (Q8 validation: watchdog fired at 2.94 GB and unloaded it). This explains the
2026-10-20 / -21 incidents. From now on, long Ollama runs go in chunks with an unload between them.
Evaluation prediction (Rule 1, local GPU, 0 CU): test set (238 tasks, 112 distinct prompts, greedy, num_predict 2048,
num_ctx 8192, added to run `bakeoff-v2-20261022` so it pairs with r5): Q4 ~26 min (760 tokens at 53 tok/s), Q5 ~30,
Q8 ~41; forgetting (300 x 2 arms, num_predict 2048): base 40-95 min, fine-tuned 15-40; in chunks of one family / 50
problems. About 3-4 h in all.
Prediction (Rule 1, local GPU, 0 CU), like-for-like comparator for the shipped Q5 (rung `r6-9b-notes-q5`, the base BF16
GGUF quantized to Q5_K_M with the same llama.cpp build): quantize ~5 min (the fine-tune's took 302 s); test set ~40 min
(r5 took 35.7 min at 52.9 tok/s, 992 tokens/answer; Q5 decodes ~0.90x as fast, 47.8 vs 53.2 tok/s for the fine-tune);
MBPP+ base arm ~33 min (Q4 base 29.7 min). ~80 min in all, chunked with an unload per chunk.

## 2026-10-22 Stage 5 built; prediction

Built by a Sonnet subagent (no existing file changed): `train/colab_train_job.py` (standalone VM job: prefetch, check,
train, gen), `train/stage5.py` (local CLI: predict, tokens, check, prepare, colab, grade, report),
`train/requirements-stage5.txt` (the exact Stage 1 stack: torch 2.12.1, transformers 5.5.0, trl 0.24.0, peft 0.21.2,
unsloth 2026.9.14), `tests/test_stage5.py` (44 tests; suite 1,082 passed + 1 skipped). Config: `unsloth/Qwen3.5-9B` @
005429ce (Apache-2.0), 16-bit LoRA r 16 / alpha 16 on the language model (attention and linear-attention projections),
lr 2e-4 cosine, warmup 0.05, batch 1 x GA 16, 2 epochs, MAX_LEN 4,608 (no row cut), loss on the final answer only
(checked on every row with the real tokenizer: labelled tokens decode to exactly the answer + `<|im_end|>`). Nested
stratified subsets 712 / 1,424 / 2,847 rows; checkpoints at 0.5, 1, 1.5, 2 epochs; each checkpoint generates one
greedy answer per distinct dev prompt (36) on the VM; grading is local on CPU. Local smoke test (4B, 4-bit, 16 rows)
ran end to end twice and found a real bug, fixed: without `prediction_loss_only=True` every eval kept the 248k-vocab
logits on the host (~1.1 GB per dev row).
Prediction (Rule 1; tokens/s derived, not measured: 312 TFLOP/s x 20 % / (6 x 9.65 B) = 1,078 tok/s; a measurement
outside ~360-3,200 tok/s is a bug): study 282 min, **25.4 CU** within Phil's 60 CU cap. Two sessions so that the first
measures throughput before the rest is paid: **session A = run r025: 76 min, 6.9 CU** (kill at 2x); session B = r050 +
r100, 225 min, 20.3 CU, re-predicted from A's measurement.
Measured, session A (18:34-19:21): `colab usage` 676.85 -> 672.90 = **3.95 CU** (predicted 6.9); wall 47 min; setup
exec 100 s (predicted 19 min); run r025 exec 41.5 min (predicted 57); training 1,691 tok/s overall, ~1,870 steady
(predicted 1,078: 1.6x faster, inside the bug band); peak train memory 19.0 GiB (25.2 reserved) of 40; dev generation
234 s per pass (~0.114 s per decode step), 7 then 2 of 36 answers cut at 2,048. The driver marked the session failed
although the run was complete: the live log is listed with no hash and the check required one. Fixed in
`train/stage5.py` (`_have`), with a regression test (45 Stage 5 tests pass).
Dev pass@1, 99 tasks (77 trained-family, 22 held-out; SE prompt-clustered):

| checkpoint | all | trained families | held-out | 5.0 subset | dev loss | probe loss |
|---|---|---|---|---|---|---|
| step 0 (untrained, same pipeline) | 45.5 (8.5) | 44.2 | 50.0 | 33.3 | 0.319 | 0.382 |
| r025 e0.5 | 63.6 (8.2) | 67.5 | 50.0 | 45.5 | 0.184 | 0.229 |
| r025 e1 | 65.7 (8.2) | 74.0 | 36.4 | 53.0 | 0.157 | 0.187 |
| r025 e1.5 | 82.8 (6.1) | 92.2 | 50.0 | 74.2 | 0.152 | 0.166 |
| r025 e2 | **82.8** (6.2) | **92.2** | 50.0 | 74.2 | 0.150 | 0.162 |

r025 e2 vs step 0: +37.4 pts, 37 tasks gained, 0 lost, sign test over prompts p = 6e-5. The gain is all in the trained
families; the held-out families (22 tasks, 8 prompts) do not move. Relevant to frozen criterion 2 (held-out gain >=
half the in-family gain) and to PLAN.md §7 question about generalization; one quarter of the data so far.
Session B prediction (measured rates: 1,691 tok/s, 0.114 s per generation step, setup kept at 19 min): runs r050 +
r100, no step-0 generation, **173 min, 15.6 CU**, kill at 346 min / 31.2 CU; Stage 5 spent 3.95 of 60.
Measured, session B (19:25-20:21): `colab usage` 672.90 -> 668.18 = **4.72 CU**; run r050 completed (178 steps,
1,804 tok/s, all 4 checkpoints downloaded), but the driver stopped the study before r100 with the same false failure.
My first fix was wrong: I misread the manifest (the `None` was its size, not its hash). Real cause: the job hashes
`log.jsonl` at the last checkpoint and appends the "done" line afterwards. Fixed at both ends: the job re-registers the
log after "done", and the driver exempts kind "log" from the hash; r025 and r050 now check complete. Lost: one run's
worth of session, about 0.5 CU of setup and the r100 slot; the r050 work is intact.
Session C prediction (r100 alone, 1,750 tok/s, 0.115 s per generation step): **110 min, 9.9 CU**, kill at 219 min /
19.7 CU; Stage 5 spent 8.67 of 60.
r050 graded locally (0 grader errors): e0.5 69.7, e1 66.7, **e1.5 80.8** (6.7), e2 74.7 % dev pass@1; dev loss keeps
falling (0.157 -> 0.134) while pass@1 does not rise: 25 % -> 50 % moves the best checkpoint 82.8 -> 80.8, inside the
noise (PLAN.md §4.3's stopping signal; loss is not the selector). Held-out split, every checkpoint of both runs:
color_segmentation 11/11 (already solved by the untrained model), **chessboard_corners 0/11** before and after training.
So far no transfer to the held-out 5.0 family: learning (N,1,k) -> (N,k) on HoughLinesP and friends has not carried
over to findChessboardCorners (PLAN.md §7 question 7; frozen criterion 2 measures exactly this). Audit of the
chessboard failures (11 tasks): step 0 = 5 "cannot unpack non-iterable numpy.int" (the 4.x (N,1,2) idiom), 3 None
returned, 3 wrong corner count; after training (r025 e2, r050 e1.5) = **0 unpacking errors**, 6 "corners not found"
(probably squares passed as the pattern size instead of inner corners), 3-5 None returned, 3 invented
`cv2.findChessboardCornersBinary` (r025). So the shape lesson did transfer to the held-out family; the answers now fail
on the pattern-size convention and on structure. Pass rate is still 0, but the failure moved.
Measured, session C (20:23-21:51): `colab usage` 668.18 -> 660.56 = **7.62 CU** (predicted 9.9); r100 complete, 356
steps, 1,749 tok/s, no false failure. **Stage 5 total 16.29 CU** of 60 (A 3.95, B 4.72, C 7.62).
r100 dev pass@1: e0.5 66.7, **e1 82.8** (6.2), e1.5 79.8, e2 66.7; dev loss 0.143 -> 0.130 -> 0.126 -> 0.124 (still
falling at e2 while pass@1 drops 16 pts: overfitting that loss does not see; probe 0.118 vs dev 0.124 at e2).

| run | rows | best checkpoint | dev pass@1 | trained fam. | held-out | 5.0 | dev loss |
|---|---|---|---|---|---|---|---|
| r025 | 712 | e2 | 82.8 (6.2) | 92.2 | 50.0 | 74.2 | 0.150 |
| r050 | 1,424 | e1.5 | 80.8 (6.7) | 89.6 | 50.0 | 71.2 | 0.136 |
| r100 | 2,847 | e1 | 82.8 (6.2) | - | - | - | 0.130 |

Convergence (PLAN.md §4.3): no doubling moves dev pass@1 beyond the noise (best vs best: -2.0 then +2.0 pts, sign test
p = 0.5 both; final vs final p >= 0.125). The report's "moves more than the noise: yes" column is wrong (it compares
|diff| with one SE and ignores its own p): to fix. Reading: data size is not the bottleneck at this scale; one epoch
on the full set is as good as two on a quarter, and a second epoch on the full set hurts.
**Chosen checkpoint: r100 @ e1** (ties r025 @ e2 at 82.8; the pre-stated tie rule takes the lower dev loss, 0.130 vs
0.150). Best-of-12 on dev inflates the score; the test set (Stage 6) is the unbiased number. Predicted vs measured CU
per epoch at 100 %: predicted 4.6 (1,078 tok/s), measured ~2.6 (1,749 tok/s, incl. evaluation). Adapters (173 MB each)
in `~/opencv-expert/stage5/s5-conv1/runs/`, not in git (Rule 8).

**Gate 4a decisions (Phil, 2026-10-22):** 1 teacher Qwen3.8-27B (Agree); 2 widen tasks with ~40 teacher-written
foundry phrasings per trainable family (n-gram-screened against dev/test) + ~30 drill tasks on the measured 5.0
changes, disjoint from the gotcha suite (Agree); 3 repair pairs by pairing a failed answer + grader message with the
same spec's verified answer, <= 1 per spec (Agree); 4 Tier A only (Agree); 5 answer format = a fenced python block with
imports and solve, no prose (Agree); 6 CHANGED: "use thinking if it increases accuracy" - test thinking mode and use
it wherever it raises the Tier-A rate, not only per CU; 7 ~3,000 examples, ~70 / 15-20 / 10 % mix (Agree); 8 CHANGED:
**cap 100 CU** for full generation, "if more CU gets a better model then let's use it" - spend for quality (thinking,
more samples, a stronger teacher where measured better), each session still predicted first and killed at 2x.

Replay data (decision 7's ~10 %): `nvidia/OpenCodeInstruct` @ 8f3ba5ba (CC-BY-4.0, checked on the HF API by me;
answers by Qwen2.5-Coder-32B-Instruct, Apache-2.0; the card says the data may be used to train models). One pinned
shard (127 MB, sha256 342757e0..., outside git); filtered to unit-test-passing, judge >= 4, no cv2, deduped; 600 rows
in `~/opencv-expert/replay/replay.jsonl` (sha256 d74d3503...), answers already single fenced python blocks; Qwen3.5
tokens: answer median 189, p99 747. Caveats: competitive-programming style prompts (low diversity, fine for
retention); attribute NVIDIA OpenCodeInstruct (arXiv:2504.04030, CC BY 4.0) if the adapter is published; check 13-gram
overlap against the MBPP+ slice when Stage 6 builds it. Rejected the runner-up (BigCode OpenRAIL-M would pass use
restrictions onto a model trained on its outputs).
Drills (decision 2b, Sonnet subagent): `foundry/drills/`, 41 training-only tasks on `gotchas.core.Check` (24 tagged
5.0, 17 timeless incl. 4 NumPy 2 drills), 122 wrong variants with pinned failure modes, 3 phrasings each; 208 tests;
full suite 1,038 passed, 1 skipped. Overlap with gotcha / dev / test / foundry prompts: task line max 0.075 (word
5-gram Jaccard; threshold 0.25), whole prompt max 0.280 (threshold 0.40). Chessboard (held out) banned from drills.
Weak spots: `drill_capture_properties` needs lint=False so it can never be Tier A (repair pairs only);
`drill_track_box` (TrackerMIL) may have low yield; `drill_bubble_count_mser` rests on the 5.0.0 MSER bug (#29652).
New measurements on the pinned stack (not fixed): after any Hershey text call in a process, a later `cv2.FontFace`
putText at a new size renders differently (1,755 of 36,000 values) - possible upstream issue; NumPy 2.4.6:
`uint8 * 587` raises OverflowError (NEP 50), `np.in1d` / `np.trapz` / `np.round_` removed; INTER_NEAREST_EXACT equals
INTER_NEAREST on only 75 of 1,890 size pairs.
Next Colab test session (before full generation): on the 58 hard-family specs (hough_segments, text_layout), compare
INT4 vs the official `Qwen/Qwen3.8-27B-FP8` (30.9 GB, fits an A100-40) and thinking on vs off; plus the phrasing
generation. Whichever raises the Tier-A rate becomes the full-generation configuration.
Prediction (Rule 1; within Phil's 100 CU cap): session s1-20261022 on an A100 with Qwen3.8-27B-INT4 @ 91bd022d:
56 phrasing prompts x 3 (70.6k tokens) + 80 think-test specs (8 per family, 12 for hough/text_layout, 16 drills) x 4
non-thinking (212.8k tokens at 867 tok/s) and x 4 thinking (1.82 M tokens at an assumed 450 tok/s, 5,700 per sample:
both unmeasured) -> 85 min, **7.6 CU** (range 5-11), kill at 170 min. Session s1fp8-20261022: the same 80 non-thinking
rows on `Qwen/Qwen3.8-27B-FP8` @ 017b9c7a (Apache-2.0, 30.9 GB, checked on the HF API) at an assumed 540 tok/s
(1.6x the INT4 bytes per token) -> 21 min, **1.9 CU**, kill at 42 min.
Measured s1 (12:29-14:14): 808 completions; load 606 s, generation 5,618 s at 707 tok/s; `colab usage` 696.73 ->
687.90 = **8.83 CU** (predicted 7.6; 105 min vs 85). Verified locally (4 workers):

| mode | Tier A (Wilson 95 %) | reasoning cut off | tokens per sample | tokens per Tier-A sample |
|---|---|---|---|---|
| non-thinking | **60.3 %** [55, 66] of 320 | 0 | 698 | 1,157 |
| thinking (12,288-token cap) | 10.6 % [8, 14] of 320 | 282 of 320 | 11,505 | 108,278 |

Per family non-thinking: contour 93.8, perspective 96.9, histogram 93.8, components 84.4, drills 64.1, homography 59.4,
hough 18.8, text_layout 12.5 %. Thinking reached an answer only on histogram (7/32 Tier A) and drills (27/64).
**Decision 6 resolved: no thinking** - it does not raise accuracy at an affordable budget; the hard families' reasoning
runs past 12k tokens on nearly every sample (Phil's condition was "if it increases accuracy").
Phrasings: 56 x 3 replies screened -> kept 40 for each of six families, 10 for text_layout (129 rejected by its content
checks); `foundry/phrasings/*.json` (250 new prompts).
s1fp8 FAILED at engine start: CUDA OOM on the A100-40 (29.8 GiB of weights and buffers in use, then a 12.25 GiB
allocation). The research's "fits with ~5 GiB spare" was wrong. The driver stopped the VM; cost 687.90 -> 687.01 =
0.89 CU. Retry once on an H100 (80 GB, native FP8): same 80 rows; H100 CU/h unknown (assume ~10) -> ~15 min, ~2.7 CU,
kill at 30 min. If it fails again, FP8 is dropped and full generation uses INT4 on the A100.
H100 retry: refused at `colab new` ("Backend rejected accelerator 'H100' ... no quota or entitlement"); no VM, 0 CU.
**FP8 dropped**; full generation = Qwen3.8-27B-INT4, non-thinking, A100. Stage 4 spend so far: 3.27 (pilot) + 8.83
(s1) + 0.89 (failed FP8) = 12.99 CU; balance 687.01.
Full generation `full1-20261022` prediction (Rule 1): 1,372 specs (7 families x 44 prompts, text_layout 14; 40
drills x 3 = 120 prompts; 398 distinct prompts), 7,188 samples (k 8 for hough_segments and text_layout); per-family
tokens per sample from s1's non-thinking run; 5.56 M output tokens at 707 tok/s (s1's measured rate) = 131 min + 10 min
load + 2 min setup = **143 min, 12.9 CU** ($1.29), kill at 286 min. text_layout is 44 % of the tokens. Output is saved
every 50 prompts and the stage is resumable.
Measured full1 (14:24-16:21): 7,188 of 7,188 completions, 28 chunks; load 603 s, generation 6,356 s; 5.60 M output
tokens at 881 tok/s (predicted 707); 161 cut at 2,048 tokens. `colab usage` 687.01 -> 676.85 = **10.16 CU** (predicted
12.9). One transient allocator OOM line at 21:10 UTC (566 MB); no chunk failed, nothing missing.
**Stage 4 Colab total: 23.15 CU** of the 100 cap (pilot 3.27, s1 8.83, failed FP8 0.89, full 10.16). Balance 676.85.
Verify (local, 4 workers, 18 min): Tier A 3,637 / 7,188 (50.6 %), 0 grader errors. By family: contour 400/420,
perspective 408/420, histogram 364/420, homography 305/428, components 274/420, hough 286/1,296 (22 %), text_layout
376/2,104 (18 %), drills 1,224/1,680 (73 %). Kept 2,225 (cap 2 per spec, 142 AST duplicates). Repair pairs by pairing:
688 (convention 271, algorithm 257, api 96, other 64).
**Dataset (Stage 4 output):** `data/train.jsonl` 2,847 rows (13.7 MB), `data/dev.jsonl` 153 rows on 34 prompts disjoint
from train (0.7 MB), `data/manifest.json`. 3,000 examples: 2,100 verified solutions (70 %), 600 repair pairs (20 %),
300 replay (10 %); 691 distinct prompts; Tier A only; no thinking-mode samples; held-out families absent; every row's
system prompt = `train.config.SYSTEM`; every final answer one fenced python block (checked by me over all rows).
Tokens under the Qwen3.5-9B template: mean 1,128, p50 1,031, **p99 3,252**, max 4,390; 0 template mismatches; every row
has exactly one response marker (repair rows train only on the final fix). p99 sets Stage 5's MAX_LEN (3,328, or
4,096 to keep every row whole). Phil reviews the doc "opencv-expert: Gate 3 Bake-off
Review" (https://claude.ai/artifact/X42uctjF9iunnWJTngcXBs): 12 decisions (fine-tune yes, base Qwen3.5-9B, Colab for
its training, imports in the contract, disjoint dev/test prompts, a 9B + notes rung, hough gain, contour TOL_C,
homography labels, the upstream issue, freezing PLAN.md §6). Apply them, re-run the greedy baselines on the fixed
prompts, record the frozen criteria here, then Stage 4. Phil, 2026-10-22 (doc comment on decision 3): train the 9B on
Colab via the Colab CLI (COMPUTE.md Path B, not installed yet; first run needs his Google sign-in in a browser); CU
prediction for his OK at Gate 4a, before any spend. Open audit: are the hough "strokes were missed" failures the
1 px strokes? For any GPU job: run `setup/commit_watchdog.ps1`, keep Ollama idle while training, and
discard the first Ollama request after a model is created before timing decode. Working method since 2026-10-17: I
orchestrate Sonnet 5.5 / Haiku 4.5 subagents (one brief and one set of files each, no git), verify every report by
re-running its tests, and integrate, log and commit myself.

Working copy of record: the WSL clone `~/Hands-On-ML` (Ubuntu-24.04). The Windows clone at
`C:\Users\ptmel\Documents\GitHub\Hands-On-ML` only syncs through GitHub.

## 2026-10-22 Gate 3 decisions (Phil: Agree on all 12; Colab for the 9B by doc comment)

Go: fine-tune Qwen3.5-9B, trained on Colab (CU prediction at Gate 4a before any spend). Before baselines freeze: the
contract asks for imports (4), ~16 phrasings per family disjoint across dev / test / foundry (5), a 9B + notes rung (6);
no thinking rung (7); hough gain 0.15 if calibration clean (8); contour TOL_C 0.80 only if no mutant slips (9);
homography mutant labels to their measured rates (10); Phil files `upstream/02` (11).

**Success criteria, frozen 2026-10-22 (PLAN.md §6, decision 12).** Measured paired on the shipped Q4_K_M GGUF against
the same base with the same notes prompt (r5: Qwen3.5-9B + notes), on the test split as rebuilt by decisions 4-5,
each judged by the exact paired sign test over distinct prompts (p < 0.05) as well as the point threshold:
1. specialization: >= +10 pts pass@1 on the test set; >= +25 pts on its 5.0 subset;
2. generalization: held-out-family gain >= half the in-family gain (underpowered: 16 held-out prompts; report its CI);
3. forgetting: <= 3-pt drop on a general Python slice (MBPP+, a few hundred problems, built at Stage 6);
4. integration: orchestrator + specialist >= orchestrator alone on the Stage 7 E2E set, with fewer orchestrator tokens.
The baseline values these are measured against come from the re-run after decisions 4-6 (entry to follow).

Applied 2026-10-22 (3 Sonnet subagents + me; every report re-checked):
- 4, 5 prompts: `to_prompt` ends "Write one function, `<sig>`, with the imports it needs. ..."; SYSTEM (train/config.py,
  PLAN.md §8.3/§8.4 templates) says the same. 16 phrasings per family: dev 0-3, test 4-11, foundry 12-15
  (`splits.foundry_phrasings`); test now has 72 distinct family prompts (+40 gotchas = 112), dev 36, none shared;
  tests pin disjointness, no near-duplicates (word-set similarity < 0.8) and no added cv2 names. `make_format_set`
  keeps to the dev phrasings. Manifests changed only in phrasing_index and prompt_sha1.
- 6: rung `r5-9b-notes` added to `bakeoff/rungs.py`.
- 8: hough `GAIN_STRONG` 0.25 -> 0.15. The five correct pipelines drift 0.0 px on 1,000 random and 600 weakest-contrast
  scenes (1 px, grey 150 on 50 -> 22 on 8). Fixed Canny caught by a 5-scene x0.15 check: (30,100) 98.5 % of gates
  (was 65.5 % at 0.25), (20,60) 67 % (1.2 %), (10,30) 0 % (fails only below ~x0.1; not calibrated). Matrix unchanged.
- Gate 2 decision 3 closed: the hough "strokes were missed" failures are not the 1 px strokes (24.6 % of 1,237 missed
  strokes vs 25.7 % of strokes; Mantel-Haenszel odds ratio 0.94 within length bins). Length predicts a miss (85 %
  missed at <= 75 px vs 37 % > 134 px): 30 of 35 replies keep only contours that approxPolyDP reduces to 2 vertices.
- 9: contour_area TOL_C 0.75 -> 0.80. Nothing slips through: pinned matrix identical; 1,000 fresh 20-scene gates, no
  mutant caught less (exact McNemar, 0 discordant); reference and pixel count 0 of 20,000 scenes rejected. Caveat:
  pixel counting's worst |err|/P creeps with sample size (0.637 at 4k scenes, 0.649 at 200k shapes, 0.689 on 1.2 M
  small gauge triangles), so 0.80 is 1.16-1.26x depending on the sample; still 0 rejections anywhere.
- 10: feature_homography mutants renamed `orb_norm_l2 (2x ORB's reject rate)`, `best_10_matches (fails 31 % of
  fresh gates)`, descriptions carry the measured rates.

Baseline re-run `bakeoff-v2-20261022` (decisions 4-6), prediction (Rule 1): greedy only, test split + gotchas, 238
tasks = 112 distinct prompts per rung (`--dedupe-greedy`), 5 rungs r1, r4, r2, r3, r5, num_predict 2048, num_ctx 8192,
4 grading workers. s/reply from Stage 3's measured generation (wall / generated replies): r1 3.9, r4 3.8, r2 12.5,
r3 20.1, r5 ~20.6 (r3 + the 1.7k-token notes prompt) -> 7 + 7 + 23 + 38 + 38 min + ~3 min of loads = **~2.0 h**,
0 CU. pass@5 is not re-run (the frozen criteria use pass@1). Watchdog running.
Incident 21:55: with the 9B loaded, 4 graders and two subagents running tests, free commit fell 5.4 -> 2.88 GB in 6
min; the updated watchdog killed the bake-off and unloaded Ollama (commit back to 18.1 GB), as designed. r1, r4, r2
were complete, r3 at 110/238. Resumed 21:56 with 2 graders and no concurrent agent jobs; the run skips done replies.
Rule from now: while a 9B rung runs, no other multi-process CPU work in WSL.
Result (done 22:54, 1,190 graded, 0 grader errors; `~/opencv-expert/bakeoff/bakeoff-v2-20261022/`, not yet copied
to `bakeoff/results/`). pass@1, all 238 / 5.0 subset / held-out, SE by prompt for all: 7B 16.8 / 10.1 / 29.5 (3.5);
4B 18.1 / 10.1 / 43.2 (3.7); 9B 32.8 / 13.4 / 43.2 (4.7); 7B + notes 21.4 / 13.4 / 38.6 (4.0);
**9B + notes 38.7 / 20.1 / 43.2 (4.9)**. 9B + notes vs 9B: +5.9 (p per prompt 0.061).
Copies in `bakeoff/results/bakeoff-v2-20261022/`.
Sandbox network wall (Rule 4 without Docker): `sandbox/run.py` now runs each child under `unshare -Un
--map-current-user` (fallback `-rn`) when the OS allows it: no IP network, no loopback, no abstract unix sockets; uid
unchanged, no capabilities. Overhead +1.9 ms per run (~2.4 %); 800 stress runs, 0 failures. Path-based unix sockets
remain reachable (only the Python guard stops them), so Docker `--network none` stays the stronger option.
`SANDBOX_NETNS=0/1` forces it off / required; `RunResult.netns` records it. Verified by me: a child that bypasses the
Python socket guard gets errno 101. Suite 769 passed.
**Frozen baselines (comparator r5-9b-notes):** test pass@1 38.7 % -> target >= 48.7 %; 5.0 subset 20.1 % -> >= 45.1 %;
held-out families 43.2 % (generalization criterion measured against this).

Stage 4 teacher pilot, approved by Phil 2026-10-21 (local time): both teachers, **hard cap 12 CU** for the pilot.
Tooling: `foundry/` (specs from the 7 trainable families' foundry phrasings and FOUNDRY seeds; offline vLLM job on the
Colab VM; local verify into Tier A/B; one repair round; yield report). Teachers (licenses and sizes checked on the HF
API by me, 2026-10-22):
- A: `RedHatAI/Qwen3.8-27B-INT4` @ 91bd022d (Apache-2.0, 19.5 GB, W4A16; base `Qwen/Qwen3.8-27B` @ 1d4bf0f2,
  Apache-2.0), non-thinking (`enable_thinking: false`), T 0.7, top_p 0.8, on an A100 (5.4 CU/h).
- B: `QuantTrio/Qwen3-Coder-30B-A3B-Instruct-AWQ` @ c58857a7 (Apache-2.0, 16.8 GB, community AWQ; base
  `Qwen/Qwen3-Coder-30B-A3B-Instruct`, Apache-2.0), T 0.8, top_p 0.95, on an L4 (1.7 CU/h).
Prediction (Rule 1; throughput is the research agent's estimate, no published L4/A100 measurement exists): each Colab
session pays ~18 min of setup (vLLM install, ~17-20 GB download, load). 200 specs x 4 samples x ~400 output tokens:
B generate 18 + 28 = 46 min = 1.3 CU, B repair (~400 prompts x 1) ~28 min = 0.8 CU; A generate ~35 min = 3.2 CU,
A repair ~25 min = 2.3 CU. **Total ~7.6 CU** (range ~5-12). Each session's exec timeout = 2x its predicted minutes
(the driver enforces it), and `colab stop` runs in a `finally`. B goes first: the cheapest place to find setup faults.
Measured, B generate (2026-10-22 01:23-01:39 local, run `~/opencv-expert/foundry/pilot-b-20261022/`): L4 (torch
2.13.0+cu130, vLLM 0.31.0, Python 3.13); 800 completions; setup + load 345 s, generation 533 s, exec 951 s;
383,486 output tokens (479 per answer) at 719 tok/s (predicted 250); KV cache 25,536 tokens. `colab usage`:
700.00 -> 699.62 = **0.38 CU** (predicted 1.3). A 3.4x miss, in the cheap direction: the throughput estimate was
scaled from other cards, and the MoE's 3.3 B active parameters decode far faster on the L4 than the bandwidth model
assumed; setup took 6 min, not 18. VM stopped, 0 active assignments.
B verify (local, 6 workers, 5.7 min, network namespace on): Tier A 140/800 (17.5 %), 101 kept (cap 2 per spec),
94 distinct. By family, all four gates: homography 70.7 %, histogram 34.8 %, components 8.0 %, contour 6.0 %,
perspective 1.7 %, text_layout 0.9 %, hough 0 %. Audited: contour_area's 104 failures are one teacher habit
(`threshold(gray, 1, ...)` on a background that is dark, not 0: one blob, "count 1 of 7 shapes"; consistent, so
metamorphic passes); components_stats' 67 are the stated order ignored; hough's are the (N,4) unpacking and missed
strokes. Teacher errors, not prompt ambiguity. 316 repair prompts.
Measured, A generate (01:40-02:02): A100-SXM4-40GB; 800 completions, 0 `<think>` leaks, 11 cut at 2,048; load 606 s,
generation 661 s; 572,375 output tokens (715 per answer) at 867 tok/s. `colab usage` 699.62 -> 697.96 =
**1.66 CU** (predicted 3.2). Pilot total so far 2.04 CU of the 12 approved.
A verify: Tier A 536/800 (67 %), 327 kept, 85.5 % of specs solved (100 % in contour, perspective, homography,
histogram, components; hough 58.6 %, text_layout 39.3 %); 0 kept examples with api findings. Hough dimming audit:
all 34 dimming failures are fixed `threshold(gray, 50/100/127)` (17 of them pass the manufactured gate), i.e. the
x0.45 / x0.15 checks are keeping fixed-threshold code out of Tier A, as designed.
Repair rounds (one sample per failed answer, T 0.6): B 316 prompts, 0.29 CU (697.96 -> 697.67), 31 to Tier A (9.8 %;
histogram 28/34, contour 0/48); A 88 prompts (only hough and text_layout failures), 0.94 CU (697.67 -> 696.73),
6 to Tier A (6.8 %). 89 of A's specs have both a failed and a Tier-A sample: 148 failed samples could pair with the
same spec's verified solution as repair examples, without a repair round.
**Pilot total: 3.27 CU** (0.38 + 1.66 + 0.29 + 0.94) of the 12 approved; predicted 7.6. Balance 696.73 CU.
Diversity: 200 specs carry 28 distinct prompts (7 families x 4 foundry phrasings), so scaling these specs mostly
resamples the same 28 tasks. ⛔ Gate 4a.

Colab: `google-colab-cli` 0.7.5 installed in `~/venvs/colab` (WSL); Phil completed the OAuth sign-in on 2026-10-22.
`colab usage`: balance 700.00 CU (COMPUTE.md assumed 200/month), rate 0.00/h, 0 assignments. Nothing spent; every
Colab job still needs a written prediction and Phil's OK for its stage (Rule 2).

## 2026-10-21 Stage 3 complete: baseline bake-off (local GPU, 0 CU)

Run `bakeoff-20261020` (19:23-00:36): 4 rungs x 238 test tasks (198 family + 40 gotchas), greedy with deduped prompts +
5 samples on one task per distinct prompt (77); 2,492 replies graded, 0 grader errors. Copies in
`bakeoff/results/bakeoff-20261020/` (report, summary, audit, config, gzipped generations and grades, hash manifest).

| rung | pass@1 all (SE by prompt) | 5.0 subset | timeless | held-out | pass@5 all | pass@1 with imports supplied |
|---|---|---|---|---|---|---|
| r1 Qwen2.5-Coder-7B | 15.5 (4.5) | 6.7 | 30.3 | 38.6 | 40.3 | 18.1 |
| r2 Qwen3.5-4B | 8.0 (3.8) | 3.4 | 15.7 | 25.0 | 35.1 | 23.1 |
| r3 Qwen3.5-9B | 32.8 (6.5) | 18.8 | 56.2 | 36.4 | 54.5 | 32.8 |
| r4 7B + notes | 21.4 (5.2) | 13.4 | 34.8 | 38.6 | 45.5 | 23.1 |

Paired pass@1 (p per task / per prompt): 9B - 7B +17.2 (7e-7 / 0.023); 9B - 4B +24.8 (1e-13 / 2e-6); notes - 7B +5.9
(0.004 / 0.18); 9B - notes +11.3 (0.003 / 0.31).
Predicted vs measured: generation 6.8 h vs 5.17 h; greedy pass 67 vs 51 min; decode 63.2 / 79.3 / 52.9 tok/s (as in
the smoke runs); tokens per reply 238 / 965 / 1,041 vs 329 / 1,202 / 1,352 (gotchas draw shorter replies).

Findings
- PLAN.md §7 question 1: convention errors dominate (T=0 failures, 7B: 98 convention, 78 algorithm; 9B: 95 / 33).
  Lever table: prior, so fine-tune. The notes move the 7B's failures from convention (98 -> 51) to algorithm (78 -> 90).
- The three families no rung passes are model failures, audited: hough_segments (4.x unpacking of 5.0's (N,4) lines,
  invented APIs), chessboard_corners (removed NumPy 2 aliases, reversed order; the notes rung raises "Chessboard not
  found" on all 22, cause not audited), text_layout (ink outside the box; the 9B returns None). Not yet audited: whether
  the hough "strokes were missed" failures are the 1 px strokes (Gate 2 decision 3).
- The contract never asks for imports ("Write one function, `def solve(...)`"): 316 replies failed on NameError cv2/np;
  all 88 of the 7B's have no import anywhere in the reply. Diagnostic re-grade with the imports prepended (official
  scores unchanged, `diag_imports.json`): 4B 8.0 -> 23.1, 7B 15.5 -> 18.1, 9B unchanged. Proposed contract fix for Gate 3.
- Truncation at num_predict 2048: 4B 120/623 replies, 9B 186/623, 7B 1, notes 0.
- Incident: free commit sat at 4-6.5 GB for most of the run and touched 2.95 GB at 23:18; the watchdog fired, killed
  nothing (its patterns covered only `train.`), and exited, so the last 78 min ran unguarded; no crash.
  `setup/commit_watchdog.ps1` now also kills bake-off jobs (resumable) and unloads Ollama models. Next long job: 4
  grading workers if the PC is in use.

## 2026-10-20 Gate 2 decisions applied; Stage 3 prepared (CPU only)

Gate 2 (Phil, results tab of the review doc): Agree on all 12 recommendations. Built by 8 subagents (6 Sonnet, 2 Haiku);
every report verified by re-running its tests or measurement.

| decision | applied as | evidence |
|---|---|---|
| 1 accept ORB-class homography | kept; tail re-check on 96,000 fresh pairs (4x) | SIFT 0 rejects; ORB-Hamming 0.183 % [0.158, 0.212] vs 0.2 % budget, stable per block; its max grows with n (70 -> 401) while p99.8 stays 59, so the budget is a rate |
| 2 Tier A = every gate | PLAN.md §4.1 | fixed threshold passes manufactured, fails only metamorphic |
| 4 stronger dimming in hough | `GAIN_STRONG = 0.25` transform + `fixed_low_canny` mutant; 52 tests | 5 legitimate pipelines: drift exactly 0.0 on 600 scenes (also at 0.2, 0.15) |
| 7 lint for swallowed specific exceptions | rule `swallowed-except` (handler of only pass/continue/break/constant; any `contextlib.suppress` call) | 258 items linted old vs new: only text_layout `swallowed_float_origin` changed; no reference flagged |
| 10 contour constants | measured, comments added | TOL_C0 unused by contourArea and by pixel count (2,000 scenes + minimal gauges) |
| 12 upstream reports | `upstream/` drafts + repros | only `bitwise_not(bool)` is worth filing (no duplicate; still on `5.x`) |
| 3, 5, 6, 8, 9, 11 | no change needed | |

Findings that need Phil at Gate 3 (no tolerance changed, Rule 3):
- hough: Canny(30,100) fails only 17.8 % of scenes at gain 0.25, so a 5-scene metamorphic gate catches it 62 % of the
  time; Canny(20,60) and (10,30) still pass. Gains 0.2/0.15 or a larger n_meta would close more of it.
- contour_area: pixel count reaches |err|/P = 0.643 on multi-shape scenes (calibration said 0.598): TOL_C = 0.75 is
  1.17x the worst case, under the 1.25x margin rule. Nothing rejected; raising TOL_C would be loosening.
- **Benchmark power and contamination.** dev and test share the same 37 prompts (4 phrasings per family, 5 for
  feature_homography); only the grading scenes differ. At T=0 identical prompts give identical code, so test pass@1
  has ~37 (+40 gotcha) independent units, not 238: the bakeoff report gives by-prompt (cluster) SEs and sign tests
  beside the binomial ones. And Stage 4 data written from these phrasings would contain the test prompts. Proposal
  for Gate 3, before the success criteria freeze: many more phrasings, disjoint between dev, test and foundry.
- Fixed now (my Stage 2 integration bug): tasks used consecutive seeds while the grader uses scenes seed..seed+19,
  so adjacent tasks shared 19 of 20 scenes. Splits now use `SCENE_STRIDE = 40`; tests pin disjoint windows and that
  no graded scene was a calibration seed. Only `seed` fields changed in dev/test; ids and prompts are unchanged.
- feature_homography mutant labels: `best_10_matches (within accepted accuracy)` fails 1.8 % of scenes, i.e. 31 % of
  20-scene gates on fresh seeds (it passes the pinned seed-0 matrix); `orb_norm_l2 (harmless here)` 0.40 %, 7.7 %.
- Process: Phil's interrupt at ~09:50 stopped the homography and splits agents before they reported; I finished both
  from their scratch outputs (tail measurements complete) and wrote the two split tests myself.

Records corrected (PLAN.md §0, dated 2026-10-20; module docstrings): chessboard corner reversal, CCL label order and
the SIFT +0.25 px offset behave identically on 4.14 (not 5.0 changes; SIFT's is the `enable_precise_upscale=False`
default, issue #23123). The MSER regression is opencv/opencv #29652, fixed on `5.x` (PR #29653), not in any release.
`remap`'s map2 reinterpretation is fixed on `5.x` (PR #30286). Builder figures that did not reproduce and are now
corrected in the modules: components border contact 66.6 % (2,000 scenes; 69 % was the test's 120), text_layout
`baseline_ignored` 20.8 % (not 27 %), chessboard window-8 dimmed drift 0.086 (not 0.077), histogram 1/(2N) at
N = 18564. Reproduced exactly: histogram 43,864 mismatches vs the 14-bit luma formula (cvtColor is the 15-bit one),
color_segmentation 2,000 scenes 0 failures.

Stage 3 preparation
- `bakeoff/` (12 modules, 79 tests): generate (Ollama /api/chat, recorded options, warm-up discarded, resumable),
  grade (process pool, can follow generation), taxonomy (format / api / lint / timeout / convention / algorithm /
  other), stats (pass@k, McNemar, cluster SEs), report, predict. Rungs r1-7b, r2-4b, r3-9b, r4-7b-notes.
- `bakeoff/notes_5_0.md`: 1,455 Qwen2.5 tokens, every line traced to the diff report, PLAN.md §0 or api_check.
- Base GGUFs in ~/opencv-expert/bases/ from the bf16 weights with the Stage 1 llama.cpp (b11408), text model only:
  4B 2.59 GiB (4.6 + 3.0 min), 7B 4.36 GiB (5.0 + 3.8 min; tokenizer from the bnb-4bit training repo, the unsloth
  bf16 repo ships none), 9B 5.38 GiB (6.5 + 6.2 min). Sizes match Stage 1's 4B and 7B exactly.
- Ollama auto-updated to 0.40.4 (was 0.32.9). Its store (15.9 GB) was copied to D:\ollama\models (C: has 17 GB free;
  the bases need 13); the app's own Model location setting overrides the OLLAMA_MODELS user variable I set.
  Phil set it at 18:28; bases registered as base-qwen2.5-coder-7b / base-qwen3.5-4b / base-qwen3.5-9b.
- Checked on 0.40.4: an explicit system message replaces the Modelfile SYSTEM (prompt_eval_count 71 with and without
  it; 19 with a 2-word system), so the harness's system message is not rendered twice; `think: false` still gives no
  thinking on both Qwen3.5 bases (75 prompt tokens = 71 + the empty think block).
- Smoke run prediction (Rule 1): dev, 1 task per family (9), greedy + 2 samples = 27 replies per rung, 4 rungs, num_ctx
  8192. Assume ~300 tokens per reply; decode 65 (7B, Stage 1), 80 (4B, Stage 1), ~45 tok/s (9B: 4B rate scaled by
  2.59/5.38 GiB, bandwidth-bound, plus some headroom) -> ~4.6 / 3.8 / 6.7 / 4.6 s per reply, ~9 min generation
  + ~1.5 min of loads. 0 CU.
- Smoke run measured (`~/opencv-expert/bakeoff/smoke-20261020/`): 17.5 min generation (pred ~10.5).

  | rung | decode tok/s (pred) | tokens/reply (pred) | truncated at 1024 | s/reply | pass@1, 9 dev tasks |
  |---|---|---|---|---|---|
  | r1-7b | 63.3 (65) | 329 (300) | 0/27 | 5.3 | 1/9 |
  | r2-4b | 79.2 (80) | 910 (300) | 17/27 | 11.7 | 0/9 |
  | r3-9b | 52.9 (45) | 845 (300) | 14/27 | 16.2 | 0/9 |
  | r4-7b-notes | 61.8 | 334 | 0/27 | 5.6 | 2/9 |

  Decode speeds are within 20 % of prediction. The miss is reply length: the Qwen3.5 bases write ~90-line solutions
  (docstrings, comments, defensive checks; no repetition loops), so 1024 tokens cuts them mid-function and the
  unclosed fence grades as a syntax error. That measures the cap, not the model: smoke 2 measures uncapped
  lengths (num_predict 4096) before the full run. Notes block measured at 1455 tokens. Some 4B NameErrors are
  real (it imported numpy inside solve() but never cv2).
- Grader bug found by the smoke run: chessboard_corners' metamorphic check raised on an empty answer on both images
  (Hausdorff distance of two empty sets). Fixed (`_set_distance` returns 0 for two empty sets; the manufactured gate
  rejects the empty answer) with a regression test; a degenerate-output fuzz of all nine families' verifiers
  (empty, zero, scalar, NaN outputs) found no other raise.
- Smoke 2 (`smoke2-20261020`, Qwen3.5 rungs, 9 dev tasks, greedy + 1 sample, num_predict 4096): median 1068 (4B) /
  1296 (9B) tokens, but 3/18 and 6/18 replies still ran to 4096. Those are reasoning written as code comments with
  thinking off, sometimes a loop (one 9B reply repeats a comment 45x; another says "I think I'm stuck in a loop").
  Every reply over ~1500 tokens failed, finished or not; the one pass was 642 tokens.
  Decision: num_predict 2048 for every rung. Mechanism: past ~1.5k tokens these bases are looping, not finishing;
  2048 changes no pass/fail in the sample and halves the time those replies cost. Truncations are reported.
- Full run design (`bakeoff-20261020`): test split + gotchas. Pass 1: greedy on all 238 tasks with `--dedupe-greedy`
  (77 distinct prompts, each reply graded on every task's own scenes: T=0 makes the repeats identical). Pass 2: 5
  samples at T=0.8 on one task per distinct prompt (77 tasks), since samples per prompt, not per task, carry the
  information. Rung order r1, r4, r2, r3. num_ctx 8192, num_predict 2048.
- Prediction (Rule 1), 462 replies per rung (77 greedy + 385 sampled): s/reply = tokens / measured tok/s + prompt
  -> r1 329/63.3 + 0.2 = 5.4 s; r4 5.7 s; r2 1202/79.2 + 0.2 = 15.4 s; r3 1352/52.9 + 0.3 = 25.9 s (token means
  under a 2048 cap, from smoke 2). Generation 42 + 44 min + 2.0 h + 3.3 h = **6.8 h** (range 5-8 h: gotcha replies
  are probably shorter), pass 1 alone ~67 min. Grading ~2,500 grades, 20-40 min of CPU overlapping generation.
  0 CU, watchdog running (14 h from 18:41). Started ~19:30, expected done ~02:30.

## 2026-10-17 Stage 2 built: benchmark, gotcha suite, removed-API gate (CPU only)

Gate 2 pre-read decisions (Phil): 1-3, 5-7 agreed; 4 changed: removed APIs get a separate gate that only reports.
Built by 10 subagents in parallel (8 Sonnet families + Sonnet gotcha suite + Haiku api_check); every report verified by
re-running its tests. Full suite after integration: **567 passed in 120 s** (OpenCV 5.0.0.93, WSL2); with the splits
below, 578. Splits (Haiku, `cvbench/splits.py`, `data/splits/`): 99 dev + 198 test tasks, 11/22 per family, both held-out
families flagged; gotcha manifest 40 rows, test only. Seed ranges DEV [3.1 M, 4 M), TEST [4 M, 5 M), FOUNDRY [5 M, 6 M)
reserved for Stage 4. The agent's grep found feature_homography's drift calibration at seeds 3,000,000-3,008,999,
overlapping my first DEV start (3 M), so I moved DEV to 3.1 M and added a test that no scored seed was used to calibrate.
Note: prompt_sha1 depends only on family and phrasing (36 distinct prompts), so prompt-level decontamination cannot
tell dev from test; seeds separate them.

| family | draft law | as built | calibration basis | tests |
|---|---|---|---|---|
| hough_segments | a + b*t endpoints | held: 1.9 + 0.7 t px; no angle term | 24,000 gauge blocks; 0 false rejects in 20,000 scenes | 47 |
| chessboard_corners (held out) | a + b*noise/contrast | REFUTED, replaced: blur-dependent bias + noise/contrast term | 3,840 blocks fit; 6,000 fresh scenes 100 % | 36 |
| perspective_rectify | a + b*blur corners; image by correlation | PARTLY: mean follows blur, worst case flat; ~7-8 px | 2,000 blocks, 8,000 corners | 32 |
| feature_homography | a + b/sqrt(N) | holds for SIFT (b 20), NOT for ORB (heavy tail); active profile sift+orb b 60 | 24,000 pairs | 32 |
| histogram_stats | exact counts, 1e-9 floats | held | all 2^24 colours; 3.16 M px | 55 |
| color_segmentation (held out) | exact areas | held | 1,200 blocks; 2,000 scenes | 42 |
| components_stats | exact; centroids 1e-6 | held | 1,200 blocks, 4,320 components | 32 |
| text_layout | containment + centring | held, per-axis (0.065 / 0.18 x ink height) + a second layer | 4,000 blocks | 31 |
Also: `cvbench/gotchas/` (40 checks: 17 in the 5.0 subset, 17 timeless, 6 controls; 88 wrong variants, 42 of them run
cleanly) 169 tests; `cvbench/api_check.py` (24-entry catalogue generated from the probe JSONs) 58 tests, integrated into
`evaluate.py` as a report-only `api` entry (local import: api_check imports Gate).

New 5.0 measurements made while building (outside the 213 probes; details in the modules and PLAN.md §0 corrections):
- bool acceptance is per function: countNonZero, findNonZero, mask= yes; moments only with binaryImage=True;
  findContours, connectedComponents*, threshold, dilate, blurs, cvtColor, add no. `bitwise_not(bool)` returns all-true
  bytes 0xFE/0xFF (silent wrong). Re-measured by me.
- findChessboardCorners returns reversed order when the top-left square is white (300/300).
- putText ignores lineType for Hershey fonts (always anti-aliased); getTextSize height = font line height.
- connectedComponents label order is not raster order at connectivity 8.
- SIFT keypoints are offset +0.246 px in x and y; ORB about -0.1.
- remap with a float32 map1 silently reinterprets a map2 of another dtype; MSER defaults give 0 regions on all 7 images
  tried; createLineSegmentDetector().detect returns (N,4); cvtColor BGR2GRAY is not the classic 14-bit formula.
- One unreproduced "double free or corruption" abort in a mixed bool/uint32/float16 script (2 of ~20 runs).
- All 213 recorded probes re-ran identically on 5.0.0.

Corrections to our own records (PLAN.md §0 and cvbench/families/README.md edited, dated 2026-10-17):
- "Each gate caught something the others missed" was an overclaim: only the manufactured (5) and metamorphic (1) gates
  catch something unique in contour_area. Tier A should be the conjunction of gates (decision for Phil).
- MSER "7 -> 0" is confounded by the changed putText fixture; the clean evidence is the texture probe 22 -> 0.
- INTER_NEAREST_EXACT changed; the wiki's "both flags equivalent, follow Pillow" is contradicted.
- The six "unlisted" removals: PR #25292 documents four; no source for GOTURN.

Process notes: one agent overwrote another's scratch file (fixed with per-family prefixes); my 2026-10-05/06 one-off
scripts vanished from the session temp folder (temp cleanup or an agent's `rm`; nothing needed lost). White papers
(3 drafts, LaTeX) are on Phil's Google Drive, kept outside the repo.

## 2026-10-06 Stage 1 complete: summary

Plumbing (QLoRA -> adapter -> merge -> GGUF Q4_K_M -> Ollama -> generate -> four-gate grade) works locally on the
RTX 3060. The default base passes, so per AGENT_PROMPT Stage 1 continues to Stage 2 (no gate).

| base | train tok/s (pred orig / revised) | peak VRAM | export | Q4_K_M | decode, warm | hydrotest |
|---|---|---|---|---|---|---|
| Qwen2.5-Coder-7B (default) | 357 (107 / 369) | 7.7 GiB | 17.1 min (from adapter, incl. 14.2 GB download) | 4.36 GiB | 64.6-65.3 tok/s (pred 50-60) | PASS: 128-token REFERENCE, 4/4 gates |
| Qwen3.5-4B | 472 (174 / 600) | 6.46 GiB | 2.3 + 8.0 min (+ llama.cpp build) | 2.59 GiB + 0.63 GiB mmproj | 78.6-84.6 tok/s | PASS non-thinking: 133 tokens, 4/4 gates |
| Qwen3.5-9B | - | - | - | - | - | dropped from local training (Phil, 2026-10-06): 7.38 GiB resident in 4-bit, ~10-10.5 GiB peak predicted vs 10.9 GiB free, 19.3 GB through WSL. Bake-off rung only |

- 7B grade at 07:31: first request decoded at 17.1 tok/s, a > 3x miss. Diagnosis: cold start of a just-created model.
  All 29/29 layers were on the GPU (11.25 GiB free at load; Ollama server.log), and two back-to-back runs right after
  decoded at 64.6 and 65.3 tok/s at full clocks (P0, 1950 MHz). Rule for the bake-off: discard the first request after
  a model is loaded before timing.
- Watchdog minimum during the export: 10.9 GB free commit (RAM free fell to 3.2 GB; the D: page file absorbed it).
- Unsloth's Modelfile for the 7B (kept as `Modelfile.unsloth`) carries the base's full Qwen2.5 Ollama template (tools,
  FIM, no `<|im_end|>` after the last assistant turn) and Qwen's default SYSTEM. For a single system + user turn it
  renders the same text as `train.config`'s TEMPLATE (the grade matched byte for byte). For Stage 6, prefer the base's
  own full template with our SYSTEM (Rule 7), and re-check it against `render_turn` for multi-turn repair dialogs.

Findings to carry forward
1. Ollama 0.32.9 thinks by default for qwen35 GGUFs; send `think: false` (`hydro_eval` does; PLAN.md §8.4 must).
2. Output format: §8.1 trains bare code, §8.4 serves JSON `{"code": ...}`. Pick one before Stage 4 (Rule 7).
3. Speed: QLoRA runs at ~69 % of the measured fp16 peak, not 20 %: a Stage 5 epoch (2.1 M tokens) on the 7B is ~1.6 h.
   Evidence for PLAN.md §7 question 6, left to Phil.
4. `save_pretrained_gguf` merges on its own; skip the separate 16-bit merge unless the merged weights are needed.
5. transformers 5.5 ignores `logging_dir`; `train_lora` sets `TENSORBOARD_LOGGING_DIR`.
6. Windows commit (~26-27 GB at idle) is the binding local constraint, not VRAM. WSL is capped at 14 GB, the D: page
   file raises the limit, and every GPU job runs under `setup/commit_watchdog.ps1`.
7. Never move or rename files a live job writes; never run Ollama while training; chains stop when a step is killed.

## 2026-10-05 Stage 1 hydrotest: setup and predictions (Phil OK'd Gate 0: mirrored networking, push, Stage 1)

Setup
- `.wslconfig` gained `networkingMode=mirrored`; WSL reaches the Windows Ollama at 127.0.0.1:11434 (version 0.32.9).
- New `train/`: `config.py` (one source for SYSTEM, chat-template rendering, Ollama TEMPLATE/PARAMETERs),
  `make_format_set.py` (64 train + 4 dev examples: 4 phrasings of `contour_area`, assistant = REFERENCE; plumbing
  only, never for real training, Rule 5), `train_lora.py` (PLAN.md §8.1 adapted: QLoRA, batch 1 x GA 16, asserts the
  base template equals `render_turn`, checks the loss mask, logs step-0 dev loss, tokens/s, peak VRAM, TensorBoard),
  `ollama_create.py` (uploads the GGUF over the Ollama API, writes the Modelfile), `hydro_eval.py` (generates one
  solution through Ollama, grades it with the four gates in the sandbox), `hydrotest.sh` (all three bases in sequence).
- tensorboard 2.21.0 added to `~/venvs/train`.

Base candidates (verified on the HF Hub, 2026-10-05)
| base | repo used | license | arch | weights |
|---|---|---|---|---|
| Qwen2.5-Coder-7B-Instruct (default) | unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit | apache-2.0 | Qwen2ForCausalLM | 5.5 GB NF4 (15.2 GB bf16 upstream) |
| Qwen3.5-4B | unsloth/Qwen3.5-4B (no pre-quantized copy exists) | apache-2.0 | Qwen3_5ForConditionalGeneration (VLM) | 9.3 GB bf16 |
| Qwen3.5-9B | unsloth/Qwen3.5-9B (no pre-quantized copy exists) | apache-2.0 | Qwen3_5ForConditionalGeneration (VLM) | 19.3 GB bf16 |

Chat templates (measured with each upstream tokenizer)
- Qwen2.5: ChatML; matches PLAN.md §8.3 exactly. Answer prefix `<|im_start|>assistant\n`.
- Qwen3.5: ChatML plus an empty `<think>\n\n</think>\n\n` before every non-thinking answer, and it trims the answer
  (the REFERENCE's trailing newline is dropped before `<|im_end|>`). The first check failed on exactly that newline;
  `render_turn` now trims for Qwen3.5 and all three bases render identically to their own templates (64/64).
  Consequence: for Qwen3.5 the `train_on_responses_only` response marker and the Ollama TEMPLATE both end with the
  empty think block, so serving matches training in non-thinking mode.
- Format finding to raise before Stage 4 (Rule 7): PLAN.md §8.4's server asks Ollama for JSON (`format: {"code": ...}`)
  but §8.1's targets are bare code. Training and serving must use one output format; decide before data generation.

Tokens: 64 examples, mean 357 tokens (Qwen2.5) / 370 (Qwen3.5), max 372; answer 127 / 133 tokens. MAX_LEN 2048 is
ample. 30 steps x 16 = 480 sequences = 7.5 passes over the 64 examples (fine for plumbing; expect train loss -> ~0).

Predictions (Rule 1), COMPUTE.md method with the measured 24.5 TFLOP/s and 20 % utilization: tok/s = 4.9e12 / (6N)
| base | N (B) | tok/s | train tokens | train min | peak VRAM | download | merge + GGUF |
|---|---|---|---|---|---|---|---|
| qwen2.5-coder-7b | 7.6 | ~107 | 171k | ~27 | ~7-8 GiB | 5.5 + 15.2 GB (merge source) | ~15 min |
| qwen3.5-4b | 4.7 | ~174 | 178k | ~17 | ~4-5 GiB | 9.3 GB | ~10 min |
| qwen3.5-9b | 9.65 | ~85 | 178k | ~35 | ~8-9 GiB | 19.3 GB | ~20 min |
Plus a one-time llama.cpp build (~5-10 min). Downloads at an assumed 20-50 MB/s. Whole chain: ~3-3.5 h wall.
Risk to the tok/s prediction: 360-token sequences at batch 1 underfill the GPU's tiles, so a miss on the slow side
is likely; it only counts as a bug beyond 3x (< 36 tok/s for the 7B). N for Qwen3.5 includes the vision tower,
which a text-only forward skips, so those two predictions lean slow.
VRAM: 10.9 GiB free with the desktop running; `hydrotest.sh` refuses to start a base while Ollama holds a model.

7B, early measurement (steps 1-3, launched 21:03 in tmux `train`)
- Load 1.6 min (5.5 GB NF4 download at ~25 MB/s). Loss mask: 128 tokens = answer + end token. Step-0 dev loss 0.5947.
- Steady 369 tok/s (step 2 -> 3: 5,715 tokens in 15.49 s), 16.5 s/step: **3.4x faster than the ~107 predicted**.
- Rule 1 diagnosis (a >3x miss in either direction): tokens are real (5,715 per step = 16 x 357, batch 1, no padding);
  loss sane (0.595 -> 0.594 -> 0.579 during warmup); implied work 6 x 7.6e9 x 369 = 16.8 TFLOP/s = 69 % of the measured
  24.5 TFLOP/s peak, so physically possible (a counting bug would show > 100 %). The miss is the assumed 20 % utilization:
  NF4 dequantization is cheap next to the matmuls at 357-token micro-batches (MLP weight 3584 x 18944: ~2 ms of matmul
  vs ~0.5 ms to dequantize 136 MB at 360 GB/s), and Unsloth overlaps activation offload with compute.
  Bears on PLAN.md §7 question 6 (expected vs measured cost): evidence for Phil, answer left to him.
- Revised predictions for the jobs not yet started, scaled by 369 x 7.6 / N (same utilization assumed for the Qwen3.5
  path, though FastModel's VLM route may be less tuned): qwen3.5-4b ~600 tok/s, 178k tokens ~5 min;
  qwen3.5-9b ~290 tok/s, ~10 min. Full Stage 5 epoch (2.1 M tokens) on the 7B: ~1.6 h, not ~5.3 h.

7B attempt 1 killed at step 9/30 (operator error, 21:08)
- transformers 5.5 ignores `SFTConfig(logging_dir=...)` (deprecated; it reads env `TENSORBOARD_LOGGING_DIR`), so the
  events went to `./runs` inside the repo. I moved that folder to `~/opencv-expert/runs/...` mid-run, assuming the writer
  kept its file descriptor; it reopens the file by path on every flush, so the next flush raised FileNotFoundError and
  the run died. No checkpoint existed yet (SAVE_STEPS=15). Lesson: never move or rename files a live job writes.
- Fix: `train_lora.py` sets `TENSORBOARD_LOGGING_DIR=$OUT/tb` before building the trainer. The partial run's events and
  summary are kept in `~/opencv-expert/runs/hydro-qwen2.5-coder-7b-attempt1/`. Its 9 steps confirm the speed:
  2:25 for 9 steps, ~362 tok/s.
- The chain continued to qwen3.5-4b as designed. A 7B rerun is queued in tmux window `rerun-7b` (`train/after.sh`),
  starting when the main log prints "hydrotest finished"; the 5.5 GB NF4 weights are cached.

qwen3.5-4b: HYDROTEST PASS (21:08-21:30)
| item | predicted | measured |
|---|---|---|
| load (incl. 9.3 GB download) | 3-8 min | 2.3 min |
| train tok/s | ~174 original / ~600 revised | 472 (177,721 tokens in 376.5 s); within 3x of both |
| peak VRAM | 4-5 GiB | 6.46 GiB reserved |
| merge 16-bit + GGUF Q4_K_M | ~10 min + llama.cpp build | 2.3 + 8.0 min (llama.cpp build included) |
| Q4_K_M size / decode | | 2.59 GiB (+ 0.63 GiB BF16 mmproj vision projector) / 76.5 tok/s |
- Loss mask 133 tokens; step-0 dev loss 0.3974 -> final 0.0001; train loss 0.0395. LoRA 32.5 M of 4.57 B params
  (language layers only). Memorized the 64 near-identical examples, as a plumbing run should.
- Generated code is byte-identical to REFERENCE; all four gates pass.
- Finding (Rule 6/7): Ollama 0.32.9 tags the qwen35 GGUF with capabilities [tools, thinking, completion] and THINKS BY
  DEFAULT in /api/chat, whatever our TEMPLATE says: the first eval generated 526 tokens (~390 of hidden thinking, then the
  code). The same model given our exact training prompt via raw /api/generate emits the 133-token answer and stops.
  With `think: false` /api/chat also returns 133 tokens, identical to raw: so non-thinking serving matches training.
  `hydro_eval.py` now sends `think: false` whenever the model reports the thinking capability; the thinking-mode
  result is kept as `hydro_eval_thinking_default.json`. PLAN.md §8.4's MCP server must also send `think: false`.
- Finding: `save_pretrained_gguf` writes its own merged 16-bit copy (8.7 GB in `gguf/`) and the GGUFs into `gguf_gguf/`,
  so the separate `save_pretrained_merged` call duplicates ~2 min and 8.7 GB per run. Keep one in Stage 6.
- Fixed `ollama_create.py`: it wrote our Modelfile before looking for Unsloth's, so an Unsloth Modelfile in the same
  directory would have been overwritten unseen. It now keeps Unsloth's as `Modelfile.unsloth` and compares TEMPLATEs.

qwen3.5-9b attempt 1: CUDA OOM while loading (21:30-21:34)
- Download (19.3 GB) completed into the shared `~/.cache/huggingface/hub/blobs` store (hub cache now 32 GB).
- `transformers.modeling_utils.caching_allocator_warmup` asked for one 7.38 GiB fp16 block and CUDA refused it with
  10.92 GiB reported free. Mechanism (hypothesis): the WSL2/WDDM paravirtualized driver cannot hand out a single block of
  that size even when the total is free; the 4B's smaller warm-up block succeeded.
- 7.38 GiB is also the measured resident size of the 4-bit 9B (NF4 language layers plus bf16 embeddings and vision
  tower), so the peak-VRAM prediction rises from ~8-9 to ~10-10.5 GiB (4B: 6.46 GiB peak = weights + ~2.6 GiB).
  That is within ~0.5 GiB of the 10.9 GiB free: training may OOM even if loading succeeds.
- Retry queued in tmux window `retry-9b`, after the 7B rerun: same run with
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (lets the allocator map the block from smaller pieces).
  Prediction: loads; trains at ~290 tok/s (~10 min) if the peak stays under 10.9 GiB, else OOM at step 1.
  If it fails again, report the 9B as not trainable on 12 GB with this stack; it stays a bake-off (inference) candidate.

21:35-21:37: 7B rerun and 9B retry both failed with CUDA OOM; idle until 22:53
- 7B rerun: OOM in the step-0 dev evaluation asking for 220 MiB, with 5.17 GiB reported free (5.73 GiB held by PyTorch).
- 9B retry (expandable_segments): the same 7.38 GiB warm-up refusal with 10.92 GiB reported free.
- My log watcher had a bug (it captured "0" twice and its failure test errored every loop), so neither failure was seen
  until it timed out at 22:52: ~75 min of idle GPU. Fixed.
- Allocation probe at 22:53 on the idle GPU (`torch.empty` + fill): largest single block 10 GiB OK; 26.25 GiB total in
  256 MiB chunks (WDDM pages CUDA allocations into shared system memory beyond the 12 GiB of VRAM, so a "fit" can
  silently become slow rather than fail). So neither the 7.38 GiB block nor the 7B's working set exceeds what the
  card can give when nothing else holds it.
- Hypothesis: interference from my own Ollama calls on the same GPU. Between 21:33 and 21:35 I loaded the 4B in the
  Windows Ollama three times to investigate the thinking finding (one with the default 5-minute keep-alive). The
  failures cluster in exactly that window, and `hydrotest.sh` only checks `/api/ps` at the start of each base; Ollama
  also returns from a `keep_alive: 0` request before its runner has released VRAM. Test: rerun with no Ollama activity.
- Rule for operators (me): no Ollama or other GPU work while a training job runs.
- Relaunched 22:55: 7B then 9B in the tmux `job` window (`bash train/hydrotest.sh qwen2.5-coder-7b qwen3.5-9b`), with
  the predictions above (7B ~369 tok/s, ~8 min of training; 9B ~290 tok/s, ~10 min, peak ~10-10.5 GiB, which may spill
  into shared memory and run slow instead of failing).

qwen2.5-coder-7b run 2: training complete, export killed by a Windows out-of-memory event (22:55-23:06)
- With no Ollama activity the 7B passed the step that OOM'd at 21:36 (supports the interference hypothesis). Its first
  three losses (0.5953, 0.5942, 0.5786) equal attempt 1's exactly: the seeded run reproduces.
| item | predicted | measured |
|---|---|---|
| train tok/s | ~107 original / ~369 revised | 356.9 (171,481 tokens in 480.4 s) |
| peak VRAM | ~7-8 GiB | 7.7 GiB reserved |
| loss | | step-0 dev 0.5947 -> final dev 0.0005; train 0.094; LoRA 40.4 M params |
- The adapter (161 MB) and summary.json were saved at 23:04. The export then began downloading the 14.2 GB bf16 base
  (the merge source) and at 23:06:15 Windows ran out of commit: 8x iaStorVD "driver failed to allocate memory",
  dwm.exe terminated "could not allocate additional memory" (23:06:30), and the WSL VM died (Hyper-V switch ports
  deleted 23:06:16). No sleep or hibernate event.
- Mechanism: commit limit 45.0 GB = 31.7 GB RAM + 13.3 GB system-managed page file on C: (C: has 18 GB free, so the
  page file cannot grow much). Windows apps already commit ~25-27 GB (measured 26.8 GB with WSL at 1.85 GB after
  restart). `.wslconfig memory=24GB` let the VM's Linux page cache (filled by the 14 GB download) grow until the sum
  passed 45 GB. My 24 GB setting was sized for the merge without checking Windows' own commit.
- Not relaunched: this affects the stability of Phil's desktop, so the fix needs his OK. Proposal: `memory=14GB` (26.8 +
  14 = 40.8 < 45) plus page-cache reclaim; close heavy apps during runs; optionally move/enlarge the page file on D:
  (a Windows setting, his to change). Prepared `train/export_gguf.py` to export the saved 7B adapter without retraining.

2026-10-06 06:24 (launched; log `hydro_run3_20261006_062423.log`; Windows commit free 17.6 GB at start, so the
14 GB VM leaves ~3.6 GB in the worst case) Phil chose: `memory=14GB` + `[experimental] autoMemoryReclaim=dropcache` (accepted without warnings by
WSL 3.0.1; the VM now shows 13 GiB), then export + grade the 7B, then one 9B attempt.
- Safeguards: a Windows-side commit watchdog (every 5 s; below 3 GB free commit it pkills `train.train_lora` /
  `train.export_gguf` in WSL and logs the minimum seen), and `hydrotest.sh` now waits up to 2 min for an idle GPU
  (empty `/api/ps` and < 2.5 GB used) before each base instead of only checking `/api/ps`.
- Predictions (Rule 1):
  - 7B export from adapter: 14.2 GB bf16 download at 25-50 MB/s (5-10 min) + merge and Q4_K_M (~10 min; llama.cpp
    already built) = ~15-20 min; WSL RAM pinned at its 14 GB cap by page cache; Windows commit free stays >= ~4 GB.
  - 7B grade: 128-token reply identical to REFERENCE, all four gates pass; decode ~50-60 tok/s (4.4 GB Q4 on 360 GB/s).
  - 9B: load from cache 3-5 min; train ~290 tok/s (~10 min) if the ~10-10.5 GiB peak fits in VRAM. If it spills into
    shared memory, tok/s drops several-fold and Windows commit falls; the watchdog stops it at 3 GB free.

06:24-06:27 result: the watchdog stopped the 7B export; I stopped the 9B
- Windows free commit: 17.6 GB at 06:24:13 -> 7.3 GB at 06:25:15 -> 2.31 GB at 06:25:46, when the watchdog killed
  `train.export_gguf` (it had loaded the adapter in 0.7 min and just begun the 14.2 GB bf16 download). Prediction
  (">= ~4 GB free") missed: 15.3 GB consumed = the 14 GB VM cap (page cache from the download fills it) + ~1.3 GB more,
  plausibly the WDDM-backed CUDA allocations of the 4-bit model.
- Two tooling faults of mine: the resume script went on to the 9B after the kill, and the watchdog had exited after
  firing, so the 9B was loading unguarded; I killed the chain by hand at 06:27 (no Windows errors this time).
  Fixed: the watchdog now kills the chain scripts before the Python jobs; `hydrotest.sh` stops the chain when a step
  exits 137/143 instead of starting the next base.
- Conclusion: with ~26-27 GB of Windows commit at idle and a 45 GB limit, this PC has ~18 GB for WSL, minus margin.
  Any step that streams a multi-GB download or checkpoint through WSL fills the VM to its cap. The 7B export (14.2 GB
  download + merge) and anything 9B do not fit safely as configured. Decision needed from Phil (page file on D:, a
  smaller VM cap with fewer apps open, or moving the export off this PC); 9B local training not recommended.

Decisions (Phil, 2026-10-06)
- Headroom: Phil adds a page file on D: (custom size, e.g. 16-48 GB). Mechanism: the commit limit is RAM + page files,
  so it rises from 45 GB to ~60-90 GB; when RAM fills, Windows pages to D: (slower) instead of failing allocations
  (the 23:06 crash). The watchdog stays on for every GPU job.
- Qwen3.5-9B: dropped from local training. It needs ~10-10.5 GiB of the 10.9 GiB free VRAM (7.38 GiB resident in 4-bit)
  and streams 19.3 GB through WSL. It stays an inference-only rung (its Q4 GGUF fits for serving) in the Stage 3
  bake-off; revisit training only if it wins there (then Colab, with Phil's OK).

07:10 7B export relaunched after the page file (log `hydro_run4_20261006_071025.log`)
- After Phil's restart: page files C: 2 GB (system managed) + D: 16 GB initial / 48 GB max; commit limit 49.7 GB and
  able to grow with the D: file; 29.7 GB free commit at launch (fewer apps open after the restart). WSL 13 GiB, GPU
  idle at 908 MiB, Ollama empty. Watchdog running (threshold 3 GB). Predictions unchanged from the 06:24 entry
  (export ~15-20 min; grade: the 128-token REFERENCE, four gates pass, ~50-60 tok/s decode).

## 2026-10-05 Stage 0 complete inside WSL2 (Gate 0)

What ran
- Phil installed WSL 3.0.1 + Ubuntu 24.04.5 (kernel 6.18.40.1), distro vhdx at `D:\WSL\Ubuntu-24.04\ext4.vhdx`, `~` has
  946 GB free. Phil installed `python3-venv python3-dev build-essential cmake libcurl4-openssl-dev` (gcc 13.3, cmake 3.28).
- `C:\Users\ptmel\.wslconfig`: `memory=24GB`, `swap=16GB` (default was 15 GB). Reason: merging a 7.6B LoRA into 16-bit
  weights for GGUF export handles ~15 GB of tensors; 15 GB of RAM left no margin.
- Cloned the branch into WSL (`~/Hands-On-ML`) from a git bundle of the Windows clone (Linux git refuses the `/mnt/c`
  repo as "dubious ownership"); `origin` points to GitHub.
- Reference env `~/venvs/cv5` (requirements-cv5.txt): `pytest` 33 passed (6.6 s); `demo_exec_vs_mms`: exec-only accepts
  7 of 8 wrong solutions, lint + manufactured + metamorphic reject 8 of 8. Both match the reference exactly.
- Training env `~/venvs/train` via `setup/train_env.sh` (log and `pip freeze` in `~/opencv-expert/logs/`):
  torch 2.12.1+cu130, unsloth 2026.9.14, unsloth_zoo 2026.9.9, transformers 5.5.0, trl 0.24.0, peft 0.21.2,
  bitsandbytes 0.50.2, xformers 0.0.35, triton 3.7.1, accelerate 1.15.0, datasets 4.3.0, opencv 5.0.0, numpy 2.4.6.
  `pytest` in this venv: 33 passed, so one venv can both train and grade.
- GPU inside WSL: RTX 3060, bf16 supported, 10.93 / 12.00 GiB free (desktop holds the rest).

Predicted vs measured
- fp16 4096^2 matmul: predicted 20-27 TFLOP/s (spec 25.6 dense, fp32 accumulate); measured 24.5 TFLOP/s.
  Consequence: Path L's peak assumption holds; at 20 % utilization the 7.6B QLoRA prediction is 24.5e12 x 0.2 / (6 x 7.6e9)
  = ~107 tok/s, so the hydrotest (30 steps x 16 x ~700 tok = 336k tok) predicts ~52 min.

Findings and open items
- Ollama 0.32.9 runs on Windows (model present: gemma4:latest, 9.6 GB). WSL uses NAT networking, so the Windows API at
  127.0.0.1:11434 is unreachable from WSL (both localhost and the host IP time out). `ollama.exe` via interop works
  (the client runs on the Windows side). Recommendation: `networkingMode=mirrored` in `.wslconfig` rather than a second
  Ollama inside WSL, because two servers would compete for the same 12 GB of VRAM.
- Before every GPU job: `ollama ps` must show no loaded model; a loaded model would take VRAM from training.
- Version watch (Rule 6): trl 0.24.0 is old relative to transformers 5.5.0; it is what unsloth 2026.9.14 resolved.
  Check `train_on_responses_only` and SFTConfig argument names against this trl at the hydrotest.
- Docker is not installed; Rule 4 bulk runs fall back to the sandbox alone until Stage 4.
- Monitoring: long jobs run detached in tmux 3.4 (`tmux attach -t train`), log to `~/opencv-expert/logs/`, with
  TensorBoard for curves.

Decisions
- The distro's vhdx lives on D: and all work happens in `~` (ext4). Mechanism: `~` is physically on D: but avoids
  9P overhead on `/mnt/d` for venvs, pytest and dataloaders. Only GGUF hand-off to the Windows Ollama uses `/mnt/d`.

## 2026-10-04 Decision: train locally on the RTX 3060, not on Colab

Phil chose the local path. Mechanistic basis: the card has 12 GB, not the <= 8 GB AGENT_PROMPT assumed, so
QLoRA on a 7B (about 6.5 GB weights + activations at batch 1, grad-accum 16, MAX_LEN 2048) fits, and a 4B fits
with headroom. The cost is speed: predicted with COMPUTE.md's formula at 25.6 TFLOP/s dense fp16 and 20 %
utilization (QLoRA dequant overhead on a 360 GB/s card): about 110 tok/s for a 7.6B, 210 tok/s for a 4B.

| stage | local prediction | notes |
|---|---|---|
| hydrotest, 30 steps x eff. batch 16 x ~700 tok (7B QLoRA) | ~50 min | use MAX_STEPS=30; proves LoRA -> merge -> GGUF -> Ollama |
| bake-off, ~800 generations per rung on Ollama Q4 | ~2 h per rung, 3-5 rungs overnight | decode ~50 tok/s for a 4.7 GB Q4 on 360 GB/s |
| teacher pilot, 200 specs x k=4 | ~4 h with a 14B coder at 4-bit (~9 GB) | 30B-A3B does not fit VRAM; try CPU offload and measure |
| full teacher generation, ~3k specs x k=4 | ~60 h at ~30 tok/s | the one stage where Colab or an API clearly pays; decide at Gate 4a |
| fine-tune 7B, 2.1 M tok/epoch | ~5 h per epoch, 2 epochs overnight | 4B: ~2.7 h per epoch |
| quant ladder evals | Q4/Q5/Q8 all fit locally | bf16 rung (15 GB) needs CPU offload or Colab |

Rules unchanged except Rule 2: the 60 CU cap is now 0 CU unless Phil re-authorizes Colab for a named stage.
Rule 1 still applies: predict tok/s and hours before every local GPU job and record the measurement.
Open question for PLAN.md §7.6: if the measured tok/s is more than 3x below ~110, suspect fp32 fallback,
no packing, or the dataloader before blaming the card.

Housekeeping: the first push of `opencv-expert-training` waited on a Git Credential Manager sign-in that was
never completed; the branch exists only locally until Phil runs `git push -u origin opencv-expert-training` once.

## 2026-10-04 Stage 0, environment survey (Windows side)

What ran
- Cloned `phil-melton/Hands-On-ML`, checked out `claude/opencv-specialist-plan` (commits 728c5a1 2026-10-03, f562f7b 2026-10-04),
  created working branch `opencv-expert-training`, copied the Rules into `CLAUDE.md`.
- Informational only, on Windows (not the reference environment): venv at `D:\opencv-expert\.venv-cv5` with
  `opencv-python-headless 5.0.0.93`, `numpy 2.4.6`, `pytest 9.1.1`. `pytest`: 12 passed, 20 failed, 1 error.
  `demo_exec_vs_mms`: every gate reports "killed: child exited with code 1 and no result".
  Mechanism: the sandbox runner relies on POSIX rlimits and process groups, so the child never starts on Windows;
  the 12 passes are the lint and contract tests that never spawn a child. This confirms the README: the kit needs WSL2.

Measured hardware (differs from AGENT_PROMPT's assumptions)
- GPU: NVIDIA GeForce RTX 3060, 12 GB VRAM (the prompt assumed <= 8 GB). Driver 595.95, CUDA 13.2 runtime visible.
  Consequence: a 7B Q4 (~4.7 GB) serves with room to spare; a 7B Q8 (~8 GB) likely fits too, so the Q8 rung of the
  quantization ladder may be evaluable locally instead of on Colab (verify: Ollama with num_ctx 8192 needs KV cache on top).
- CPU: i7-12700 (20 threads). RAM: 31.7 GB.
- Disk: C: 16.8 GB free (below the ~40 GB the prompt asks for), D: 931 GB free, G: 16 GB free.
  Decision: put the WSL distro, venvs, datasets and GGUFs on D:.
- Ollama 0.32.9 is installed on Windows. Docker: not installed (Rule 4 bulk-run isolation falls back to the sandbox alone
  unless Docker Desktop or Docker inside WSL is added).
- WSL: not installed (`wsl --status`: "The Windows Subsystem for Linux is not installed").
- git: credential helper = manager (Git for Windows), user phil-melton; remote reachable.

Predicted vs measured: no GPU spend yet (0 CU).

Decisions
- Work in WSL2, as the prompt requires; do not port the sandbox to Windows (it would change the thing under test).
- Keep the D: drive for everything large; WSL2 distro install target must be D: as well (`wsl --import` or move the vhdx).
