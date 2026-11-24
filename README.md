# Project Lipsy

Project Lipsy is a Minecraft 1.8.9 PotPvP bot built around a question: can a bot learn to fight like an expert by watching one play? PotPvP is a form of melee PvP where players use splash potions to heal mid-fight. Lipsy uses **behavior cloning** to learn from my gameplay, from how I aim and strafe to when I stop attacking to heal.

## Why I built it

I used to play a lot of Minecraft PvP and was a top-tier PotPvP player. My GitHub handle, `Br1d3e`, is my old Minecraft name. I thought it would be cool to teach a bot how I played as an expert. The part that still interests me most is whether it can learn *when* to act, not just how to press the right keys once it has decided.

## How it works

Lipsy uses **hierarchical imitation learning** to separate combat mechanics from decisions about what to do next:

- **Low-level skills** handle aiming, movement, clicking, and blockhitting (briefly blocking with the sword during combat). Separate models learn aim, strafing, W-tap timing, and blockhitting. Clicking replays recorded human timing patterns selected by heuristics rather than using a trained model.
- **High-level decisions** predict one of six states: `attack`, `heal`, `pearl`, `speed`, `eat`, or `refill`. These correspond to fighting, throwing a healing potion, using an ender pearl, taking a speed potion, eating, and restocking the hotbar. The current model uses a shared encoder with a six-state prediction head and auxiliary heads for signals such as heal and pearl starts.
- **A rule-based arbitrator** in the Forge client decides what to execute. It can reject actions the bot cannot perform, let an action already underway finish, or override a prediction when the bot needs an emergency heal or runs out of hotbar potions.

## Recording and training

I record my fights with the Minecraft 1.7.10 Forge module in [`recorder-1.7.10`](recorder-1.7.10/). It writes one JSON record per game tick to `frames.jsonl.gz`, the compressed form of `frames.jsonl`. Each frame includes player observations such as motion, aim yaw and pitch, inventory, and health, plus observations about the opponent. A separate event log captures discrete game events.

The Python code in [`models`](models/) turns recordings and action labels into training datasets for the low-level skill models and the high-level decision model. The [high-level dataset builder](models/hl_decision/hl_decision_datasets.py) pairs observations through tick `t` with the expert's state and action-start labels at tick `t+1`. That distinction matters: a tick spent healing is not necessarily a decision to start healing.

## Inference

The [`bot-1.8.9`](bot-1.8.9/) Forge client sends observations to the Python [socket server](models/socket_server.py) over TCP. The server loads trained checkpoints, keeps recent observation windows, and returns skill predictions and high-level scores. The client applies its rules and executes the chosen behavior in Minecraft.

## Current milestone and challenges

The trained low-level models are working as intended in my tests: the bot imitates my combat mechanics well enough to put up a fight. High-level decisions are the current focus. I tried separate binary triggers for individual actions on `hl/binary-trigger`, then moved to a shared encoder with multiple prediction heads on `hl/multi-head`.

Healing and refilling basically work with help from the arbitrator, but their timing still falls short of the demonstrations. Pearling is unsolved: in my runtime tests, the deployed model has never selected `pearl` as its high-level decision.

My leading hypothesis is distribution shift. On recorded expert trajectories, the model appears better at recognizing an action already underway than at initiating a heal or pearl from the situations the bot actually reaches. That's not a confirmed cause. Current work separates action-start metrics from state accuracy, compares checkpoints selected for different objectives, and traces model predictions, executed states, and rule overrides during play.

## Project structure

| Path | Role                                                                                          |
| --- |-----------------------------------------------------------------------------------------------|
| [`recorder-1.7.10/`](recorder-1.7.10/) | Forge recorder, target feature tracking, and gameplay labeling tools                          |
| [`models/`](models/) | Dataset builders and training code for aim, strafe, W-tap, blockhit, and high-level decisions |
| [`models/hl_decision/`](models/hl_decision/) | High-level datasets, multihead training, and earlier trigger experiments                      |
| [`models/socket_server.py`](models/socket_server.py) | TCP inference server for the trained models                                                   |
| [`bot-1.8.9/`](bot-1.8.9/) | Forge runtime, combat skills, and high-level arbitration                                      |
| [`dev_log.txt`](dev_log.txt) | Early development notes, no longer updated                                                    |
