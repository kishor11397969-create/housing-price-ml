# Curve Quest: playtest and feedback evidence

**Source of this feedback:** Claude (an AI assistant made by Anthropic), as suggested by the course instructor. It is **AI feedback plus automated test runs, not human peer testing**, and it is labelled that way on purpose. The in-game "Playtest log" tab is for real peers; no peer entries have been recorded yet, so none are shown here.

## 1. Automated playtest (real runs of the game)

A script opened `index.html` in a headless browser, played the levels with different decisions, and copied the game's own feedback and scores. No JavaScript errors occurred in any run.

| Level | Player decision | What the game reported | Stars |
|---|---|---|---|
| 1 Straight-line trap | Degree 1 (too simple) | "Underfitting": training error already high (0.264); 9% of best possible | 0 |
| 1 Straight-line trap | Degree 3 | "Well balanced": test error 0.017, 98% of best possible | 3 |
| 2 The memoriser (12 points) | Degree 11 (too flexible) | "Overfitting": training error 0.072 but test error 999+; 0% of best | 0 |
| 2 The memoriser (12 points) | Degree 3 | "Well balanced": test error 0.239; 100% of best | 3 |
| 3 Data is power | Degree 10 with 10 points | "Overfitting": training error 0.021, test error 999+; 0% | 0 |
| 3 Data is power | Degree 10 after buying 24 more points | Still overfitting but much better: test error 25.2 (down from 999+) | 0 (1%) |
| 3 Data is power | Degree 4 | Score 76% of best | 2 |
| 4 Rein it in | Degree 12, no penalty | "Overfitting": training error 0.000, test error 433.5; 0% | 0 |
| 4 Rein it in | Degree 12, penalty strength 4 | "Well balanced": test error 0.371; 79% of best | 2 |
| 4 Rein it in | Degree 3, no penalty | Test error 0.357; 92% of best | 3 |

**What this shows (and does not show).**
- The player's decisions change the outcome, and the results match the theory: too simple underfits (high bias), too flexible overfits (training error near 0, test error huge), more data reduces the damage (999+ down to 25.2), and a regularisation penalty rescues a degree-12 model (433.5 down to 0.371).
- This is evidence that the game mechanics and the concepts are correct. It is **not** evidence that human learners learned the concepts.

## 2. Claude's feedback on the game

**Strengths**
- Real computation behind everything (least-squares polynomial fitting with a ridge penalty on freshly generated noisy data); nothing is hard-coded.
- The learner discovers the idea by experimenting with sliders, and each submit uses a new hidden test sample, so the test set cannot be memorised.
- Clear feedback after each submit, an error-curve chart revealed after playing, 5 levels covering bias, variance, sample size, regularisation and the trade-off.
- Learning objectives and a how-to-play guide are built into the game.

**Suggestions for improvement**
1. Level 3: degree 4 shows the message "Underfitting" while still earning 2 stars; the wording and the star could be aligned to avoid confusion.
2. Level 4: in this test, a plain degree-3 model (92%) beat degree 12 with a penalty (79%). The level goal asks the player to beat the low-degree model, so the penalty range or data could be tuned so that regularisation can win.
3. Test errors above 999 are displayed as "999+", which hides how large the overfit is; showing the training/test error ratio could help.
4. Entries in the playtest log are saved only in the browser, so they must be exported (the CSV at the bottom of the tab) into the repository.

## 3. Human peer playtests (to be added)

No human peer playtests have been recorded yet. When peers play, use the Playtest log tab and paste its CSV here.
