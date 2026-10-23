"""Stage 7 end-to-end (E2E) evaluation: 20 multi-step tasks, a grader and a headless runner.

    tasks.py    the 20 tasks (manufactured from the cvbench families, exact truths, derived tolerances)
    compose.py  geometry, rendering and tolerance-propagation helpers the tasks share
    grade.py    grade a work dir's answer.json against a task's truth
    run.py      build / run / grade / report / predict CLI around ``claude.exe -p``
"""
