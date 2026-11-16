"""
Homework 2 - A/B Testing
Author: Lyudmila Zakaryan

What this file does
-------------------
I have four advertisements. In the code I call each of them a "bandit",
like a slot machine in a casino. When I show an advertisement I get a
reward, which is just a number. On average the four bandits pay 1, 2, 3
and 4, but every single reward has noise in it, so from a few tries I can
not tell which one is the best.

I have 20000 tries. I want to collect as much reward as I can, but I do not
know which bandit is the good one. So all the time I have to decide between
two things:
    explore = try some other bandit, maybe it is better than I think
    exploit = stay with the bandit that looks best so far

I solve this in two ways and then compare them:
    1. Epsilon-Greedy, where epsilon goes down as 1/t
    2. Thompson Sampling for normal rewards with known precision

After the run you will find:
    rewards.csv   every reward of every trial (Bandit, Reward, Algorithm)
    plots/        four pictures
    and the cumulative reward and regret in the log messages

To run it:
    pip install -r requirements.txt
    python Bandit.py

The template asked to use the logger instead of print(), so there is no
print() anywhere in this file.
"""
############################### LOGGER
import csv
import os
from abc import ABC, abstractmethod

import matplotlib

# With "Agg" the pictures go straight into files and no window pops up.
# I chose it so the script does not hang waiting for somebody to close a window.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger


############################### SETTINGS
# I keep all the numbers of the experiment here at the top.
# If I want to try something else, I change it in one place only.

# True average reward of each bandit. Given in the homework.
BANDIT_REWARDS = [1, 2, 3, 4]

# How many pulls each algorithm gets. Given in the homework.
NUMBER_OF_TRIALS = 20000

# Precision is 1 / variance, it says how noisy the rewards are.
# We could choose it ourselves. I took 1, so the standard deviation is also 1.
# That is the same as the gap between two neighbour bandits (1 and 2, 2 and 3...),
# so the bandits overlap a lot and the task is not too easy.
PRECISION = 1.0

# Epsilon at trial t is EPSILON / t.
# Trial 1 -> 1.0 (always explore), trial 2 -> 0.5, trial 100 -> 0.01 and so on.
EPSILON = 1.0

# Thompson Sampling starts with a guess about the mean of every bandit.
# The guess is a normal distribution around 0, and this number is its precision.
# 0.01 is a very small precision, so the guess is very wide (standard deviation 10).
# It is my way of saying "I have no idea yet what the mean is".
# At first I had 1.0 here and it worked badly. The guess was too tight around 0,
# so bandit 4 almost never looked interesting and was ignored for about 600 trials.
PRIOR_PRECISION = 0.01

# With a fixed seed the random numbers are the same on every run,
# so anybody who runs my file gets exactly my results.
SEED = 42

# My colours: pink, brown and sky blue (plus a light brown, because
# there are four bandits and I need a fourth colour).
PINK = "#EC5E9A"
BROWN = "#7B4B2A"
SKY_BLUE = "#56B4E9"
LIGHT_BROWN = "#C8A27C"

# One colour for each bandit. The best bandit (number 4) gets the pink.
BANDIT_COLOURS = [LIGHT_BROWN, BROWN, SKY_BLUE, PINK]

# One colour for each algorithm, used when I compare the two.
ALGORITHM_COLOURS = {"EpsilonGreedy": PINK, "ThompsonSampling": SKY_BLUE}

# The results are saved next to this file. I build the path from the place of
# the file, so it does not matter from which folder the script is started.
BASE_FOLDER = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_FOLDER, "rewards.csv")
PLOTS_FOLDER = os.path.join(BASE_FOLDER, "plots")


class Bandit(ABC):
    """The abstract bandit from the template.

    Nobody can create an object of this class. It is only a list of the
    methods that a real bandit class must have. If I forget one of them in
    a child class, Python gives an error when I try to create the object.
    """
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self, p):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def experiment(self):
        pass

    @abstractmethod
    def report(self):
        # store data in csv
        # print average reward (use f strings to make it informative)
        # print average regret (use f strings to make it informative)
        pass

#--------------------------------------#


def build_results(algorithm, bandits, chosen_bandits, rewards, estimates_history):
    """Collect everything from one experiment into one dictionary.

    Both algorithms need the same numbers for the report and for the plots.
    I did not want to write that code two times, so it lives here.

    The regret is also calculated here. Regret of one trial means: how much
    I lost because I did not pull the best bandit. It is the mean of the
    best bandit minus the mean of the bandit I pulled. If I pulled the best
    one, the regret of that trial is zero.

    :param algorithm: name of the algorithm, "EpsilonGreedy" or "ThompsonSampling"
    :type algorithm: str
    :param bandits: the bandit objects after the experiment
    :type bandits: list
    :param chosen_bandits: which bandit (0, 1, 2, 3) was pulled in every trial
    :type chosen_bandits: numpy.ndarray
    :param rewards: the reward of every trial
    :type rewards: numpy.ndarray
    :param estimates_history: estimated mean of every bandit after every trial,
                              one row is one trial and one column is one bandit
    :type estimates_history: numpy.ndarray
    :returns: all results of the experiment
    :rtype: dict
    """
    true_means = np.array([bandit.p for bandit in bandits], dtype=float)

    # true_means[chosen_bandits] gives the true mean of the pulled bandit
    # for all trials at once, so I do not need a loop here.
    regrets = true_means.max() - true_means[chosen_bandits]

    return {
        "algorithm": algorithm,
        "bandits": bandits,
        "true_means": true_means,
        "chosen_bandits": chosen_bandits,
        "rewards": rewards,
        "regrets": regrets,
        # cumsum is a running total: place t holds the sum of trials 1..t
        "cumulative_rewards": np.cumsum(rewards),
        "cumulative_regrets": np.cumsum(regrets),
        "estimates_history": estimates_history,
        # bincount counts how many times each bandit number appears
        "pull_counts": np.bincount(chosen_bandits, minlength=len(bandits)),
    }


def save_and_log_report(results):
    """Write the rewards to the CSV file and log the main numbers.

    The CSV has the three columns from the homework: Bandit, Reward, Algorithm.
    One row is one trial. I open the file in append mode ("a"), so the second
    algorithm puts its rows under the rows of the first one and at the end
    all rewards are in one file.

    :param results: the dictionary made by :func:`build_results`
    :type results: dict
    :returns: nothing
    :rtype: None
    """
    algorithm = results["algorithm"]
    number_of_trials = len(results["rewards"])

    # The header line should be written only once, when the file is still new.
    file_is_new = not os.path.exists(CSV_PATH)

    try:
        with open(CSV_PATH, "a", newline="") as csv_file:
            writer = csv.writer(csv_file)
            if file_is_new:
                writer.writerow(["Bandit", "Reward", "Algorithm"])
            for bandit_index, reward in zip(results["chosen_bandits"], results["rewards"]):
                # Python counts from 0, but I call the bandits 1, 2, 3, 4. That is why +1.
                writer.writerow([bandit_index + 1, round(float(reward), 4), algorithm])
        logger.info(f"[{algorithm}] {number_of_trials} rewards were saved to {CSV_PATH}")
    except OSError as error:
        # If the file can not be written I still want to see the numbers below,
        # so I only log the problem and go on.
        logger.error(f"[{algorithm}] could not write the CSV file: {error}")

    # [-1] is the last element, so it is the total after all trials.
    cumulative_reward = results["cumulative_rewards"][-1]
    cumulative_regret = results["cumulative_regrets"][-1]

    logger.info(f"[{algorithm}] Cumulative reward after {number_of_trials} trials: {cumulative_reward:.2f}")
    logger.info(f"[{algorithm}] Cumulative regret after {number_of_trials} trials: {cumulative_regret:.2f}")
    logger.info(f"[{algorithm}] Average reward per trial: {cumulative_reward / number_of_trials:.4f}")
    logger.info(f"[{algorithm}] Average regret per trial: {cumulative_regret / number_of_trials:.4f}")

    # What the algorithm thinks about each bandit at the end (this calls __repr__).
    for bandit in results["bandits"]:
        logger.info(f"[{algorithm}] {bandit}")

    # A small check: is the bandit that was pulled the most really the best one?
    favourite = int(np.argmax(results["pull_counts"]))
    best = int(np.argmax(results["true_means"]))
    if favourite != best:
        logger.warning(
            f"[{algorithm}] The most pulled bandit is number {favourite + 1}, "
            f"but the best one is number {best + 1}. The algorithm got stuck on a wrong bandit."
        )


class Visualization():
    """Makes the plots of the homework and saves them as PNG pictures."""

    def __init__(self, all_results, plots_folder=PLOTS_FOLDER):
        """Keep the results that will be drawn.

        :param all_results: one results dictionary for each algorithm
        :type all_results: list
        :param plots_folder: folder for the pictures
        :type plots_folder: str
        """
        self.all_results = all_results
        self.plots_folder = plots_folder
        # Make the folder if it is not there yet. No error if it already exists.
        os.makedirs(self.plots_folder, exist_ok=True)

    def _save(self, figure, file_name):
        """Save a figure into the plots folder and close it.

        :param figure: the figure to save
        :type figure: matplotlib.figure.Figure
        :param file_name: name of the picture, for example "plot.png"
        :type file_name: str
        :returns: nothing
        :rtype: None
        """
        path = os.path.join(self.plots_folder, file_name)
        figure.tight_layout()
        figure.savefig(path, dpi=150)
        plt.close(figure)
        logger.info(f"Plot saved: {path}")

    def plot1(self):
        """Show how each algorithm learns the bandits.

        I make one picture for each algorithm. Each picture has two parts,
        the same data with a normal x axis on the left and a log x axis on
        the right.

        A solid line is what the algorithm thinks the mean of a bandit is.
        The dashed line in the same colour is the real mean. When the
        algorithm learns, the solid line goes to its dashed line.

        I need the log axis because almost everything interesting happens
        in the first hundred trials. On the normal axis that part is so
        small that you can not see it.

        :returns: nothing
        :rtype: None
        """
        # Visualize the performance of each bandit: linear and log
        for results in self.all_results:
            algorithm = results["algorithm"]
            history = results["estimates_history"]
            trials = np.arange(1, len(history) + 1)

            figure, axes = plt.subplots(1, 2, figsize=(14, 5))
            for axis, scale in zip(axes, ["linear", "log"]):
                for index, true_mean in enumerate(results["true_means"]):
                    colour = BANDIT_COLOURS[index % len(BANDIT_COLOURS)]
                    axis.plot(trials, history[:, index], color=colour, linewidth=1.8,
                              label=f"Bandit {index + 1} estimate "
                                    f"(pulled {results['pull_counts'][index]} times)")
                    axis.axhline(true_mean, color=colour, linestyle="--", alpha=0.7)
                axis.set_xscale(scale)
                axis.set_title(f"{algorithm}: learning process ({scale} scale)")
                axis.set_xlabel("Trial")
                axis.set_ylabel("Estimated mean reward")
                axis.grid(alpha=0.3)
            axes[0].legend(fontsize=8)
            self._save(figure, f"plot1_learning_{algorithm}.png")

    def plot2(self):
        """Put the two algorithms on the same picture.

        Left is the cumulative reward, there higher is better.
        Right is the cumulative regret, there lower is better. When an
        algorithm has found the best bandit it stops losing reward, so its
        regret line becomes flat.

        :returns: nothing
        :rtype: None
        """
        # Compare E-greedy and thompson sampling cummulative rewards
        # Compare E-greedy and thompson sampling cummulative regrets
        figure, (reward_axis, regret_axis) = plt.subplots(1, 2, figsize=(14, 5))

        for results in self.all_results:
            trials = np.arange(1, len(results["rewards"]) + 1)
            colour = ALGORITHM_COLOURS.get(results["algorithm"], BROWN)
            reward_axis.plot(trials, results["cumulative_rewards"], color=colour,
                             linewidth=2, label=results["algorithm"])
            regret_axis.plot(trials, results["cumulative_regrets"], color=colour,
                             linewidth=2, label=results["algorithm"])

        reward_axis.set_title("Cumulative reward")
        reward_axis.set_ylabel("Sum of all rewards so far")
        regret_axis.set_title("Cumulative regret")
        regret_axis.set_ylabel("Sum of all regrets so far")
        for axis in (reward_axis, regret_axis):
            axis.set_xlabel("Trial")
            axis.grid(alpha=0.3)
            axis.legend()
        self._save(figure, "plot2_cumulative_reward_and_regret.png")

#--------------------------------------#

class EpsilonGreedy(Bandit):
    """One advertisement for the Epsilon-Greedy algorithm.

    How Epsilon-Greedy decides: it throws a kind of coin. With probability
    epsilon it explores and pulls a random bandit. The rest of the time it
    exploits and pulls the bandit that has the best average so far.

    Here epsilon is 1/t, where t is the number of the trial. In the
    beginning I know nothing, so it is good to explore a lot. Later I
    already know the good bandit, and every random pull only costs reward,
    so epsilon should be small.
    """

    def __init__(self, p):
        """Create one bandit.

        :param p: the true mean reward of this bandit, the algorithm is not allowed to look at it
        :type p: float
        """
        self.p = p              # the real mean, only pull() uses it to make rewards
        self.p_estimate = 0.0   # what I think the mean is, I start from zero
        self.N = 0              # how many times this bandit was pulled

    def __repr__(self):
        """Describe the bandit in one line.

        :returns: the true mean, my estimate and the number of pulls
        :rtype: str
        """
        return (f"EpsilonGreedy bandit: true mean = {self.p}, "
                f"estimated mean = {self.p_estimate:.3f}, pulls = {self.N}")

    def pull(self):
        """Pull the bandit one time.

        randn() gives a number from the standard normal distribution
        (mean 0, standard deviation 1). I divide it by sqrt(PRECISION) to
        get the noise I want and then add the true mean of the bandit.

        :returns: one noisy reward
        :rtype: float
        """
        return np.random.randn() / np.sqrt(PRECISION) + self.p

    def update(self, x):
        """Take a new reward and correct my estimate of the mean.

        My estimate is simply the average of all rewards of this bandit.
        I do not store the old rewards. The new average can be found from
        the old one:
            new average = old average + (new reward - old average) / N
        So the estimate moves a little bit in the direction of the new
        reward, and the more pulls I have, the smaller this step is.

        :param x: the reward I just got from this bandit
        :type x: float
        :returns: nothing
        :rtype: None
        """
        self.N += 1
        self.p_estimate = self.p_estimate + (x - self.p_estimate) / self.N

    @classmethod
    def experiment(cls, bandit_rewards=BANDIT_REWARDS, number_of_trials=NUMBER_OF_TRIALS,
                   epsilon=EPSILON):
        """Run the Epsilon-Greedy experiment from the first trial to the last.

        I made it a classmethod because the experiment is not about one
        bandit. It creates all four bandits and plays with them together.

        :param bandit_rewards: true mean reward of each bandit
        :type bandit_rewards: list
        :param number_of_trials: how many pulls are allowed
        :type number_of_trials: int
        :param epsilon: the starting epsilon, trial t uses epsilon / t
        :type epsilon: float
        :returns: all results, see :func:`build_results`
        :rtype: dict
        """
        bandits = [cls(p) for p in bandit_rewards]
        number_of_bandits = len(bandits)

        # Here I write down what happened in each trial. I need it later
        # for the CSV file and for the plots.
        chosen_bandits = np.zeros(number_of_trials, dtype=int)
        rewards = np.zeros(number_of_trials)
        estimates_history = np.zeros((number_of_trials, number_of_bandits))
        explore_count = 0

        for t in range(1, number_of_trials + 1):
            # epsilon gets smaller with every trial
            current_epsilon = epsilon / t

            # random() gives a number between 0 and 1.
            # It is below epsilon exactly with probability epsilon.
            if np.random.random() < current_epsilon:
                # explore: any bandit, all have the same chance
                index = np.random.randint(number_of_bandits)
                explore_count += 1
            else:
                # exploit: the bandit with the highest estimate
                index = int(np.argmax([bandit.p_estimate for bandit in bandits]))

            # pull the chosen bandit and learn from what it gave
            reward = bandits[index].pull()
            bandits[index].update(reward)

            # t starts from 1 but the arrays start from 0, so t - 1
            chosen_bandits[t - 1] = index
            rewards[t - 1] = reward
            estimates_history[t - 1] = [bandit.p_estimate for bandit in bandits]

        logger.debug(f"[EpsilonGreedy] explored {explore_count} times out of {number_of_trials} trials")
        return build_results("EpsilonGreedy", bandits, chosen_bandits, rewards, estimates_history)

    @classmethod
    def report(cls, results):
        """Save the rewards to the CSV file and log the reward and the regret.

        :param results: what :meth:`experiment` returned
        :type results: dict
        :returns: nothing
        :rtype: None
        """
        save_and_log_report(results)

#--------------------------------------#

class ThompsonSampling(Bandit):
    """One advertisement for the Thompson Sampling algorithm.

    Epsilon-Greedy keeps one number for each bandit, the average.
    Thompson Sampling keeps more: for each bandit it has a belief about the
    mean, and the belief is a normal distribution with
        m        the middle of the belief, my best guess of the mean
        lambda_  the precision of the belief, it shows how sure I am

    In each trial every bandit draws one random number from its belief, and
    I pull the bandit whose number is the biggest.

    Why this works: a bandit that I pulled only a few times has a wide
    belief, so sometimes it draws a big number and gets one more chance.
    A bandit that I pulled many times has a narrow belief and draws numbers
    very close to its real mean. So the exploring happens by itself, and it
    stops by itself when I am sure. I do not need an epsilon.

    Known precision means that I know how noisy the rewards are (PRECISION,
    in the formulas it is tau). Only the means are unknown.
    """

    def __init__(self, p):
        """Create one bandit.

        :param p: the true mean reward of this bandit, the algorithm is not allowed to look at it
        :type p: float
        """
        self.p = p                      # the real mean, only pull() uses it
        self.tau = PRECISION            # the known precision of the rewards
        self.m = 0.0                    # middle of my belief, I start at zero
        self.lambda_ = PRIOR_PRECISION  # very small at the start = I am not sure at all
        self.sum_x = 0.0                # sum of all rewards of this bandit
        self.N = 0                      # how many times this bandit was pulled

    def __repr__(self):
        """Describe the bandit in one line.

        :returns: the true mean, my estimate and the number of pulls
        :rtype: str
        """
        return (f"ThompsonSampling bandit: true mean = {self.p}, "
                f"estimated mean = {self.m:.3f}, pulls = {self.N}")

    def pull(self):
        """Pull the bandit one time.

        :returns: one noisy reward, normal with mean p and variance 1 / tau
        :rtype: float
        """
        return np.random.randn() / np.sqrt(self.tau) + self.p

    def sample(self):
        """Draw one random guess of the mean from my belief.

        It looks like pull(), but pull() uses the real mean and the real
        noise, and here I use my belief: middle m and precision lambda_.
        When lambda_ is big the guess is almost equal to m.

        :returns: one guess of the mean of this bandit
        :rtype: float
        """
        return np.random.randn() / np.sqrt(self.lambda_) + self.m

    def update(self, x):
        """Take a new reward and update my belief about this bandit.

        This is the Bayes rule for normal rewards with known precision.
        It comes down to two lines:
            new lambda = old lambda + tau
            new m      = tau * (sum of all rewards) / new lambda
        The first line says that every reward makes me a bit more sure.
        The second line is close to the usual average of the rewards.
        It is this short because my starting belief has its middle at 0,
        so the starting belief adds nothing to the sum on top.

        :param x: the reward I just got from this bandit
        :type x: float
        :returns: nothing
        :rtype: None
        """
        self.N += 1
        self.sum_x += x
        self.lambda_ += self.tau
        self.m = self.tau * self.sum_x / self.lambda_

    @classmethod
    def experiment(cls, bandit_rewards=BANDIT_REWARDS, number_of_trials=NUMBER_OF_TRIALS):
        """Run the Thompson Sampling experiment from the first trial to the last.

        :param bandit_rewards: true mean reward of each bandit
        :type bandit_rewards: list
        :param number_of_trials: how many pulls are allowed
        :type number_of_trials: int
        :returns: all results, see :func:`build_results`
        :rtype: dict
        """
        bandits = [cls(p) for p in bandit_rewards]
        number_of_bandits = len(bandits)

        chosen_bandits = np.zeros(number_of_trials, dtype=int)
        rewards = np.zeros(number_of_trials)
        estimates_history = np.zeros((number_of_trials, number_of_bandits))

        for t in range(number_of_trials):
            # every bandit draws a guess from its belief, the biggest guess wins
            index = int(np.argmax([bandit.sample() for bandit in bandits]))

            # pull the winner and update only its belief
            reward = bandits[index].pull()
            bandits[index].update(reward)

            chosen_bandits[t] = index
            rewards[t] = reward
            estimates_history[t] = [bandit.m for bandit in bandits]

        logger.debug(f"[ThompsonSampling] finished {number_of_trials} trials")
        return build_results("ThompsonSampling", bandits, chosen_bandits, rewards, estimates_history)

    @classmethod
    def report(cls, results):
        """Save the rewards to the CSV file and log the reward and the regret.

        :param results: what :meth:`experiment` returned
        :type results: dict
        :returns: nothing
        :rtype: None
        """
        save_and_log_report(results)




def comparison(all_results, plots_folder=PLOTS_FOLDER):
    """Compare the two algorithms with one more picture and with numbers.

    I wanted to answer two questions.

    Left part: where did the pulls go? The bars show how many times each
    algorithm pulled each bandit. A good algorithm gives nearly all pulls
    to bandit 4.

    Right part: how fast did it learn? The line is the average reward so
    far (cumulative reward divided by the trial number). The dashed line
    at 4 is the best average that is possible. The sooner a line comes
    close to it, the faster the algorithm learned.

    The same things are also written to the log as numbers.

    :param all_results: one results dictionary for each algorithm
    :type all_results: list
    :param plots_folder: folder for the picture
    :type plots_folder: str
    :returns: nothing
    :rtype: None
    """
    # think of a way to compare the performances of the two algorithms VISUALLY and
    os.makedirs(plots_folder, exist_ok=True)
    figure, (bar_axis, line_axis) = plt.subplots(1, 2, figsize=(14, 5))

    true_means = all_results[0]["true_means"]
    number_of_bandits = len(true_means)
    positions = np.arange(number_of_bandits)
    # the bars of the two algorithms stand next to each other, so each one is thinner
    bar_width = 0.8 / len(all_results)

    for number, results in enumerate(all_results):
        algorithm = results["algorithm"]
        colour = ALGORITHM_COLOURS.get(algorithm, BROWN)
        trials = np.arange(1, len(results["rewards"]) + 1)

        bars = bar_axis.bar(positions + number * bar_width, results["pull_counts"],
                            width=bar_width, color=colour, label=algorithm)
        bar_axis.bar_label(bars, fontsize=8)

        line_axis.plot(trials, results["cumulative_rewards"] / trials, color=colour,
                       linewidth=2, label=algorithm)

    bar_axis.set_xticks(positions + bar_width * (len(all_results) - 1) / 2)
    bar_axis.set_xticklabels([f"Bandit {i + 1}\n(true mean {mean:g})" for i, mean in enumerate(true_means)])
    # Bandit 4 has about 20000 pulls and the others only a few.
    # Without the log scale the small bars can not be seen at all.
    bar_axis.set_yscale("log")
    bar_axis.set_ylabel("Number of pulls (log scale)")
    bar_axis.set_title("How often was each bandit pulled?")
    bar_axis.legend()

    line_axis.axhline(true_means.max(), color=BROWN, linestyle="--", linewidth=2,
                      label=f"Best possible average ({true_means.max():g})")
    line_axis.set_xscale("log")
    line_axis.set_xlabel("Trial (log scale)")
    line_axis.set_ylabel("Average reward so far")
    line_axis.set_title("Running average reward")
    line_axis.grid(alpha=0.3)
    line_axis.legend()

    path = os.path.join(plots_folder, "comparison.png")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    logger.info(f"Plot saved: {path}")

    # Now the same in numbers.
    best_bandit = int(np.argmax(true_means))
    for results in all_results:
        trials_total = len(results["rewards"])
        share_best = results["pull_counts"][best_bandit] / trials_total
        logger.info(
            f"[Comparison] {results['algorithm']}: "
            f"cumulative reward = {results['cumulative_rewards'][-1]:.2f}, "
            f"cumulative regret = {results['cumulative_regrets'][-1]:.2f}, "
            f"best bandit pulled in {share_best:.2%} of the trials"
        )

    # the winner is the one that lost less, so the one with the smaller regret
    winner = min(all_results, key=lambda results: results["cumulative_regrets"][-1])
    logger.info(f"[Comparison] Lower regret in this run: {winner['algorithm']}")


if __name__=='__main__':

    np.random.seed(SEED)

    # The CSV is opened in append mode. If I do not delete the old file,
    # the rows of the previous run stay in it and the file grows on every run.
    if os.path.exists(CSV_PATH):
        os.remove(CSV_PATH)
        logger.debug(f"Old file removed: {CSV_PATH}")

    logger.info(f"Bandit rewards: {BANDIT_REWARDS}, trials: {NUMBER_OF_TRIALS}, "
                f"epsilon: {EPSILON}/t, precision: {PRECISION}, seed: {SEED}")

    # first the two experiments
    logger.info("Running the Epsilon-Greedy experiment...")
    epsilon_greedy_results = EpsilonGreedy.experiment()

    logger.info("Running the Thompson Sampling experiment...")
    thompson_results = ThompsonSampling.experiment()

    # then the report: CSV file, cumulative reward and cumulative regret
    EpsilonGreedy.report(epsilon_greedy_results)
    ThompsonSampling.report(thompson_results)

    # then the plots
    all_results = [epsilon_greedy_results, thompson_results]
    visualization = Visualization(all_results)
    visualization.plot1()
    visualization.plot2()

    # and my own comparison at the end
    comparison(all_results)

    logger.info("Done. Everything finished without errors.")
