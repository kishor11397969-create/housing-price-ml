# Playtest and feedback evidence

**Source of feedback for this submission:** Claude (an AI assistant made by Anthropic), as suggested by the course instructor. This file is honest about what that is: it is **AI feedback and automated test runs, not human peer testing.** If human peers test the game later, their results can be added in the section at the bottom.

## 1. Automated playtest (real runs of the game, round 1 of each case)

A script opened `index.html` in a headless browser and played each strategy **6 times** (the game generates fresh random data each time). Scores are the game's own 0-100 score, copied from the game screen.

| Case | Player decision | Scores in 6 runs | Average |
|---|---|---|---|
| 1 Sampling | Station intercepts, 30 or 150 interviews | 0 in every run | 0 |
| 1 Sampling | Open online poll, 30 or 150 interviews | 0 in every run | 0 |
| 1 Sampling | Random draw, 30 interviews | 52, 49, 48, 47, 48, 46 | 48.3 |
| 1 Sampling | Random draw, 150 interviews | 78, 76, 76, 77, 76, 78 | 76.8 |
| 1 Sampling | Stratified draw, 30 interviews | 69, 68, 68, 68, 68, 67 | 68.0 |
| 1 Sampling | Stratified draw, 150 interviews | 86, 86, 85, 86, 85, 84 | 85.3 |
| 3 Overfitting | Degree 1 (too simple) | 49, 68, 82, 67, 64, 64 | 65.7 |
| 3 Overfitting | Degree 3 | 97, 100, 97, 100, 100, 100 | 99.0 |
| 3 Overfitting | Degree 9 (too flexible) | 0, 0, 0, 34, 91, 0 | 20.8 |
| 4 Causation | No controls | 0 in every run | 0 |
| 4 Causation | Control for temperature (the confounder) | 92, 97, 98, 95, 87, 99 | 94.7 |
| 4 Causation | Control for all variables | 99, 93, 87, 95, 98, 86 | 93.0 |

**What this shows (and does not show).**
- The player's decision really changes the outcome, and the scores follow the data science ideas: a biased design (station, poll) scores 0 **even with 5x more interviews**, so more data does not fix bias; stratified beats random at the same budget; a degree-9 model overfits (score collapses, and is very unstable between runs) while a moderate model does best; ignoring the confounder gives 0.
- This is evidence that the game mechanics and scoring work as designed. It is **not** evidence that human learners learned, because a script cannot learn.

## 2. Claude's feedback on the game

**Strengths**
- Real computation behind every answer (random sampling, regression, polynomial fitting); no stored answers.
- Feedback explains the concept right after the decision (for example the bias vs variance message in Case 1).
- Three rounds per case with harder data, and the first attempt is the learning score, so retrying is for exploring.
- Includes How to play, Objectives, Key terms, a Learning report and a Playtest tab.

**Suggestions for improvement**
1. In Case 4 round 1, controlling for irrelevant variables (wind, weekday) is not penalised (93 vs 94.7). A small penalty or a message would teach "do not control for everything."
2. Playtest notes and progress are stored only in the browser, so they are not in the repository. Export them into this file.
3. The fonts load from Google; offline the game uses fallback fonts and still works.
4. In `l4Analyse` there is a leftover placeholder variable (`naive`) that is never used; it can be deleted.

## 3. Human peer playtests (optional, to be added)

No human peer playtests have been recorded yet. If peers play the game, add their real scores and comments here.
