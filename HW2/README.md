# HW2 - A/B Testing with Multi-Armed Bandits

**Author: Lyudmila Zakaryan**

Four advertisements (bandits) with true mean rewards `[1, 2, 3, 4]` and 20000 trials.
I used two algorithms that try to find the best advertisement and collect as much reward as possible on the way:

- **Epsilon-Greedy** with epsilon decaying as `1/t`
- **Thompson Sampling** for Gaussian rewards with known precision

## How to run

```bash
pip install -r requirements.txt
python Bandit.py
```

The run takes a few seconds. The random seed is fixed, so the results are the same on every run.

## Files

| File | What it is |
|---|---|
| `Bandit.py` | All the code: `Bandit` (abstract), `EpsilonGreedy`, `ThompsonSampling`, `Visualization`, `comparison()` |
| `requirements.txt` | Packages needed to run the code |
| `rewards.csv` | Created by the script. One row per trial: `Bandit, Reward, Algorithm` (40000 rows) |
| `plots/plot1_learning_EpsilonGreedy.png` | Learning process of Epsilon-Greedy (linear and log scale) |
| `plots/plot1_learning_ThompsonSampling.png` | Learning process of Thompson Sampling (linear and log scale) |
| `plots/plot2_cumulative_reward_and_regret.png` | Cumulative reward and cumulative regret of both algorithms |
| `plots/comparison.png` | Extra comparison: pulls per bandit and running average reward |

## My choices

| Setting | Value | Why |
|---|---|---|
| Epsilon | `1/t` | Asked in the homework. First trial is 100% exploring, then it drops fast. |
| Precision of rewards (`tau`) | `1` | Reward noise has standard deviation 1, the same as the distance between two neighbour bandits, so the task is not trivial. |
| Prior of Thompson Sampling | mean `0`, precision `0.01` | A wide starting belief ("I do not know the mean yet"). With precision `1` the best bandit was not tried for hundreds of trials. |
| Seed | `42` | Makes the results reproducible. |
| Colours of the plots | pink, brown, sky blue | My own palette. In the comparison plots Epsilon-Greedy is pink and Thompson Sampling is sky blue. |

## Results (seed 42)

| Algorithm | Cumulative reward | Cumulative regret | Best bandit pulled |
|---|---|---|---|
| Epsilon-Greedy | 79734.47 | 135.00 | 99.37% of trials |
| Thompson Sampling | 79925.13 | 42.00 | 99.83% of trials |

Regret in one trial = (mean of the best bandit) - (mean of the bandit that was pulled).
The best possible total reward is about 4 x 20000 = 80000, so both algorithms are close to it in this run.

What the plots show:

- **Epsilon-Greedy** found bandit 3 first and stayed on it for about 120 trials. It moved to bandit 4 only when a random exploration step happened to try it. Epsilon `1/t` gets small very fast, so there were only 9 exploration steps in 20000 trials.
- **Thompson Sampling** tried every bandit in the first trials (because of the wide starting belief), saw that bandit 4 is the best and then almost never left it. Bandits 1 and 2 were dropped after two or three pulls, bandit 3 after 29 pulls.

## BONUS - a better implementation plan

One run is not enough to judge an algorithm, so I repeated the experiment with 100 different seeds:

| Algorithm | Median regret | Mean regret | Worst regret | Runs stuck on a wrong bandit |
|---|---|---|---|---|
| Epsilon-Greedy (`1/t`) | 32 | 2310 | 20357 | 12 of 100 |
| Thompson Sampling | 29.5 | 30 | 54 | 0 of 100 |

So Epsilon-Greedy with `1/t` is usually fine, but in about 1 run out of 8 it stays on a wrong advertisement until the end.
The reason: with `1/t` it explores only about 10 times in total. If the best bandit is not tried in these few steps, or its first reward is unlucky, the algorithm never comes back to it.

Based on this, my suggestions are:

1. **Slower decay of epsilon.** Use `epsilon = min(1, c / t)` with a bigger `c` (for example 50), or `1 / sqrt(t)`, or keep a small floor like `0.01`. More exploring at the start costs a little reward but removes the risk of getting stuck.
2. **Force one pull of every bandit at the start** (or use optimistic starting estimates, for example 10 instead of 0). Then no bandit is judged before it was tried.
3. **Report the average of many runs, not one run.** Run each algorithm with many seeds and plot the mean regret with a band around it. One run can be lucky or unlucky, like the table above shows.
4. **Do not assume the precision is known.** In real advertising we do not know the noise. Thompson Sampling can learn the mean and the precision together (Normal-Gamma prior), or use a Beta prior if the reward is click / no click.
5. **Add UCB1 as a third algorithm.** It needs no random exploring and no tuning, so it is a good baseline for the comparison.
6. **Allow rewards that change in time.** Advertisements get old. A sliding window or a discount on old rewards lets the algorithm follow the change.
7. **Stop early.** With Thompson Sampling we can compute the chance that each bandit is the best one. When this chance is above 95% the experiment can stop and the budget is saved.
8. **Code structure.** Keep settings in a separate config file, vectorize the loops with NumPy to run many repetitions fast, and add small unit tests for `pull()` and `update()`.
