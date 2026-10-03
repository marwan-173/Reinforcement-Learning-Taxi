# Reinforcement Learning Taxi --- Reward Design with Generalized Prioritized Sweeping

## Authors

-   **Marwan Zayyad**
-   **Baraah Dahle**

## Project Overview

This project studies how **reward design**, **Prioritized Sweeping**,
and **kernel-based state generalization** affect reinforcement learning
performance in the **Gymnasium Taxi-v4** environment.

The baseline learning algorithm is **SARSA**. It is extended with a
planning mechanism based on Prioritized Sweeping and a similarity kernel
that transfers updates to related Taxi states.

The project includes:

-   SARSA baseline learning
-   Prioritized Sweeping
-   Kernel-based Generalization
-   Five reward functions
-   Reward design comparison
-   Hyperparameter sweeps
-   Ablation studies
-   Experiments across 5 random seeds
-   95% confidence intervals
-   Automatic plot generation
-   Saved learned policies
-   Policy evolution and best-policy animations

------------------------------------------------------------------------

## Environment

The project uses **Gymnasium Taxi-v4**, a discrete reinforcement
learning environment in which a taxi must:

1.  Navigate to the passenger.
2.  Pick up the passenger.
3.  Navigate to the destination.
4.  Drop off the passenger successfully.

The environment contains **500 states** and **6 possible actions**.

------------------------------------------------------------------------

## Learning Methods

### SARSA

SARSA is used as the baseline on-policy temporal-difference learning
algorithm.

### Prioritized Sweeping

Observed transitions are stored in a model and assigned priorities based
on their TD error. Important previous transitions are replayed during
planning so that useful information can propagate faster.

### Kernel-based Generalization

The kernel extends an update from the current state to similar Taxi
states.

Two states are considered for generalization only when they have the
same:

-   Passenger status/location
-   Destination

Similarity is then determined using the Manhattan distance between taxi
positions. Only the closest neighboring states are updated.

### Full Algorithm

The full method combines:

**SARSA + Prioritized Sweeping + Kernel Generalization**

This allows the agent to learn from real interaction, reuse important
previous experience, and transfer useful updates to related states.

------------------------------------------------------------------------

## Reward Designs

Five reward functions are evaluated.

### 1. Base Reward

Uses the original Taxi-v4 reward without modification.

### 2. Sparse Reward

Provides reward only for successful pickup and drop-off events, creating
a much weaker learning signal.

### 3. Custom Reward 1 --- Fast Delivery

Encourages the agent to complete the task quickly by rewarding
successful delivery with an additional time-dependent bonus.

### 4. Custom Reward 2 --- Efficient Driving

Uses stronger penalties for illegal actions and attempts to move into
walls, encouraging safer and more efficient behavior.

### 5. Custom Reward 3 --- Progress Reward

Provides intermediate feedback based on whether the taxi moves closer to
the passenger before pickup or closer to the destination after pickup.

------------------------------------------------------------------------

## Experiments

### Reward Comparison

The five reward designs are compared under the same learning setup to
study how the quality and density of the reward signal affect learning.

The experiments showed that reward design strongly influences
convergence. In the reported experiments, **Custom Reward 3 (Progress
Reward)** provided the fastest improvement and shortest episodes, while
the sparse reward learned much more slowly because of its limited
feedback.

### Ablation Study

The ablation study compares four configurations:

-   SARSA
-   SARSA + Prioritized Sweeping
-   SARSA + Kernel Generalization
-   Full Algorithm: SARSA + Prioritized Sweeping + Kernel

The study is performed using both the Base Reward and Custom Reward 1.

The results show that both planning and generalization contribute to
learning performance, while combining them produces the strongest
overall behavior in the reported experiments.

### Hyperparameter Sweeps

Two parameters are systematically evaluated:

**Learning rate (`alpha`)**

``` text
0.05, 0.10, 0.20, 0.40
```

**Exploration rate (`epsilon`)**

``` text
0.01, 0.05, 0.10, 0.20, 0.30
```

Both sweeps are performed using:

-   Base Reward
-   Custom Reward 1

------------------------------------------------------------------------

## Experimental Setup

The default configuration is located in:

``` text
src/config.py
```

Main settings:

``` text
Training episodes:      1000
Random seeds:           5
Learning rate (alpha):  0.1
Discount factor (gamma): 0.95
Exploration (epsilon):  0.1
Planning steps:         5
Kernel beta:            4.0
Kernel c:               0.5
Kernel neighbors:       5
Maximum episode steps:  200
```

Results are aggregated across **5 seeds**, and comparison plots use a
**20-episode rolling mean** with **95% confidence intervals**.

------------------------------------------------------------------------

## Main Results

### Reward Design Comparison

#### Cumulative Reward

![Reward Comparison - Cumulative
Reward](results/comparison_plots/reward_comparison_reward.png)

#### Episode Length

![Reward Comparison - Episode
Length](results/comparison_plots/reward_comparison_episode_length.png)

### Ablation Study

#### Base Reward

![Base Reward
Ablation](results/comparison_plots/base_ablation_reward.png)

#### Custom Reward 1

![Custom Reward 1
Ablation](results/comparison_plots/custom1_ablation_reward.png)

These figures are generated automatically from the experiment CSV files.

------------------------------------------------------------------------

## Policy Animations

The project also includes GIF animations that visualize the learned
policies.

Example full-model policies:

### Base Reward

![Base Reward Full
Policy](results/animations/best_policies/base_reward_full.gif)

### Custom Reward 1

![Custom Reward 1 Full
Policy](results/animations/best_policies/custom_reward_1_full.gif)

A separate policy-evolution animation shows how the learned behavior
changes during training.

------------------------------------------------------------------------

## Project Structure

``` text
RL_Project/
│
├── src/
│   ├── __init__.py
│   ├── agent.py
│   ├── config.py
│   ├── environment.py
│   ├── kernel.py
│   ├── main.py
│   ├── metrics.py
│   ├── planning.py
│   └── rewards.py
│
├── experiments/
│   ├── ablation_study.py
│   ├── experiment_runner.py
│   ├── parameter_sweeps.py
│   ├── plot_results.py
│   ├── policy_animation.py
│   ├── report_plots.py
│   ├── reward_comparison.py
│   └── run_all.py
│
├── results/
│   ├── animations/
│   ├── comparison_plots/
│   ├── csv/
│   ├── plots/
│   └── policies/
│
├── tests/
│   ├── __init__.py
│   ├── test_agent.py
│   ├── test_kernel.py
│   ├── test_metrics.py
│   ├── test_planning.py
│   └── test_rewards.py
│
├── Presentation.pptx
├── Report.pdf
├── README.md
└── requirements.txt
```

------------------------------------------------------------------------

## Installation

Create a virtual environment:

``` bash
python -m venv .venv
```

Activate it on Windows:

``` bash
.venv\Scripts\activate
```

Install the required packages:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## Configuration

Experiment settings can be changed in:

``` text
src/config.py
```

For a quick test, use:

``` python
NUM_SEEDS = 1
```

For the complete experiments used in the report:

``` python
NUM_SEEDS = 5
```

------------------------------------------------------------------------

## Running the Project

### Run the Complete Experimental Pipeline

Recommended:

``` bash
python -m experiments.run_all
```

This automatically runs:

1.  Reward comparison
2.  Parameter sweeps
3.  Ablation study
4.  Individual plot generation
5.  Report comparison plot generation

### Train a Single Agent

``` bash
python -m src.main
```

### Run Reward Comparison

``` bash
python -m experiments.reward_comparison
```

### Run Parameter Sweeps

``` bash
python -m experiments.parameter_sweeps
```

### Run Ablation Study

``` bash
python -m experiments.ablation_study
```

### Generate Individual Plots

``` bash
python -m experiments.plot_results
```

### Generate Report Comparison Plots

After the required experiment CSV files have been generated:

``` bash
python -m experiments.report_plots
```

### Generate Policy Animations

After running the ablation study or the complete pipeline:

``` bash
python -m experiments.policy_animation
```

------------------------------------------------------------------------

## Output

Generated files are organized as follows:

``` text
results/csv/               Raw experiment results
results/plots/             Individual experiment plots
results/comparison_plots/  Comparison figures used in the report
results/policies/          Learned policies and checkpoints
results/animations/        Policy and training-evolution animations
```

------------------------------------------------------------------------

## Report and Presentation

The repository includes the complete project documentation:

-   [Full Project Report](Report.pdf)
-   [Project Presentation](Presentation.pptx)

The report contains the full methodology, experimental analysis,
hyperparameter evaluation, ablation study, results, and conclusions.

------------------------------------------------------------------------

## Reproducibility

The final experiments use **5 random seeds** to reduce dependence on a
single random run.

The plotting scripts aggregate results across seeds and visualize both
the mean learning behavior and uncertainty using 95% confidence
intervals.

For faster debugging or testing, the number of seeds can temporarily be
reduced in `src/config.py`.

------------------------------------------------------------------------

## Technologies

-   Python
-   Gymnasium
-   NumPy
-   Pandas
-   Matplotlib
-   Seaborn
-   SciPy
-   tqdm

------------------------------------------------------------------------

## Course Project

This project was developed as part of a **Reinforcement Learning**
course and focuses on experimentally studying how reward design,
model-based planning, state generalization, and hyperparameter selection
influence learning in the Taxi-v4 environment.
