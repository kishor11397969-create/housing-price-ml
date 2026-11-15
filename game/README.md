# Part B: Data Detective Casebook

A browser game where the learner **discovers** four data science concepts by making decisions and seeing the computed consequences. It is one file, with no installation and no server.

**To play:** download this repository, open `game/index.html` in Chrome, Edge or Firefox. (Internet is only needed for the Google Fonts; the game also works with fallback fonts.)

## Concepts taught

| Case | Concept | What the player does | What the game computes live |
|---|---|---|---|
| 1. Survey Desk | Sampling and bias | Chooses a sampling method (station, open poll, random, stratified) and a sample size | Draws a real sample from a generated population, then repeats the design 400 times to show bias vs spread |
| 2. Property Desk | Outliers and leverage | Inspects sales, spends limited paperwork checks, removes records | Re-fits a regression line after every removal and scores the slope against the true one |
| 3. Forecast Lab | Overfitting, bias-variance | Chooses polynomial degree, can hold out validation data | Fits polynomial models, reports training / validation / fresh-test error |
| 4. Policy Desk | Correlation vs causation | Ticks which variables to control for | Runs a regression with those controls and compares the effect with the true causal effect |

## Learning objectives

After finishing the game a learner can:

**Case 1 (Sampling)**
- Tell selection bias (the centre is wrong) apart from sampling variance (the spread is wide).
- Choose a design that removes bias, then a sample size that controls spread.
- Explain why stratifying helps when groups differ or one group is small.

**Case 2 (Outliers)**
- Separate a data error from a genuine extreme value.
- Use evidence about cause, not distance from the line, to decide what to remove.
- See how one high-leverage point can bend a regression line.

**Case 3 (Overfitting)**
- Explain why training error keeps falling as a model gets more flexible.
- Use held-out data to estimate error on unseen cases.
- Describe the bias-variance trade-off, and why small data demands simpler models.

**Case 4 (Causation)**
- Recognise a confounder and adjust for it to reveal the real effect.
- Explain why adjusting for a collider or a mediator can mislead.
- Judge what would really happen if a policy changed the first variable.

## Game design (the learning loop)

1. **Brief:** each round opens a client problem with a generated dataset (seeded random numbers, so every play is different).
2. **Decision:** the player picks a method, removes records, sets model flexibility or chooses controls.
3. **Computation:** the decision is run on the data immediately. No answers are stored.
4. **Feedback:** charts, numbers and a plain explanation of the concept.
5. **Progression:** three rounds per case, each harder (smaller budgets, more noise, trickier traps). Finishing a case unlocks the next.
6. **Scoring:** 0-100 per round. Only the **first attempt** counts as the learning score, so retries are for exploring.

The **Learning report** tab shows first-try scores per round, and the **Playtest** tab records peer feedback.

## Implementation notes

- Plain HTML + JavaScript, no libraries. Charts are drawn as SVG.
- Real computation: random sampling, ordinary least squares, Legendre polynomial fitting, multiple regression with controls.
- Progress is saved in the browser (`localStorage`).
- "Teacher mode: unlock all cases" on the Play page opens every case for marking.

## Playtest evidence

See [`PLAYTEST.md`](PLAYTEST.md): automated playtest results and Claude feedback.
