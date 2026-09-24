# PLAN.md — Stage 1: Image Classification From Scratch (NumPy only)

## Goal

Build logistic regression, softmax regression, and a 2-layer neural network **from scratch in NumPy** and train them on Fashion-MNIST.

**The agent writes all the code.** The human (Jani) directs the work, runs and checks the results, and must understand every piece well enough to explain it in an interview. Working code alone is not the finish line; understanding is. Each phase therefore ends with an understanding gate the agent must run before moving on.

Stage 1 is part of a longer project (Stage 2: PyTorch CNN on CIFAR-10, Stage 3: packaging and tests). Do not start those unless asked.

## Roles

- **Agent:** writes all code, runs experiments, teaches, and quizzes the human.
- **Human:** decides direction, reviews every diff, runs the verification checks (gradient check, tiny-subset overfit), answers the understanding gates, and writes a short note after each phase.

### Agent rules

1. **Work on one phase at a time.** Do not implement later phases. Stop at the end of each phase and wait.
2. **Explain before you code.** Before each phase, give a plain-language explanation of the math and the plan (5-10 sentences, no unexplained jargon). Assume the human is at CS229 lecture 8 and new to ML.
3. **Keep the code readable.** Small functions, clear names, comments that say *why*, and shape comments on every array operation (e.g. `# (N, 784) @ (784, 10) -> (N, 10)`). No clever one-liners the human cannot follow.
4. **After writing, walk the human through it.** For each new file, explain what each function does and how it maps to the math from the explanation.
5. **Run the understanding gate** at the end of each phase (see below). Do not mark a phase done until the human has answered in their own words. If an answer is wrong or vague, re-explain differently and ask again.
6. **Never invent numbers.** Any metric that goes into the README or notes must come from a run the human saw. If you have not run it, say so.
7. **Keep the split clean.** The test set is touched exactly once, at the very end of Phase 5. All tuning uses the validation set. Flag any violation immediately.
8. **Report problems honestly.** If a check fails or results look suspicious (e.g. accuracy near chance), say so plainly and debug it with the human instead of hiding it.
9. **One commit per task** with a clear message. Do not commit large data files.
10. **Ask, don't assume,** when a requirement is ambiguous.

## Constraints

- Python 3.10+
- Allowed libraries: `numpy`, `matplotlib`, `pytest`, and the standard library (`urllib`, `gzip`, `struct`, `argparse`, `json`, etc.)
- **Not allowed for model code:** `torch`, `tensorflow`, `sklearn`, `jax`, `autograd`, or any library that provides models, losses, or automatic differentiation.
- Fixed random seed for reproducibility, set in one place.
- Everything must run on a laptop CPU in a few minutes or less. Use vectorized NumPy, no per-sample Python loops in training.

## Dataset

Fashion-MNIST: 70,000 grayscale 28x28 images, 10 clothing classes.
- Source: the raw IDX files from the official Zalando Research repo (github.com/zalandoresearch/fashion-mnist), downloaded with `urllib` and parsed with `gzip` + `numpy`. Do not use torchvision/keras loaders.
- Split: 60,000 train file → **50,000 train / 10,000 validation** (fixed seed, shuffled). The 10,000-image official test file is **held out until the end**.

## Repo structure

```
.
├── PLAN.md
├── README.md
├── notes/                 # human's own write-ups, one file per phase
├── src/
│   ├── data.py            # download, load, normalize, split
│   ├── gradcheck.py       # numerical gradient checker
│   ├── logistic.py        # binary logistic regression
│   ├── softmax.py         # softmax regression
│   ├── mlp.py             # 2-layer neural network
│   ├── metrics.py         # accuracy, confusion matrix
│   └── plots.py           # loss curves, sample grids
├── experiments/           # scripts that call src/ and save results
├── results/               # plots and JSON of metrics (small files only)
└── tests/
```

## Understanding gate (end of every phase)

The agent asks the human 3-4 questions, one at a time, and waits for answers in the human's own words. The human then writes `notes/phase_N.md` with these headings:

1. What was built
2. What I expected vs. what happened
3. What confused me
4. How I would explain this to an interviewer in 30 seconds

Questions should test understanding, not recall. Example: "If we removed the max-subtraction in softmax, what could go wrong, and on what kind of input?"

## Phases

Each phase ends with a **Done when** checklist plus the understanding gate.

### Phase 0 — Setup and data

Tasks:
- Create the repo structure and `requirements.txt` (numpy, matplotlib, pytest).
- `data.py`: download and cache the IDX files, load into NumPy arrays, scale pixels to [0, 1], flatten to 784-dim vectors, create the 50k/10k split with a fixed seed.
- `plots.py`: save a grid of sample images with labels.

Done when:
- [ ] Shapes printed and correct: train (50000, 784), val (10000, 784), test (10000, 784).
- [ ] Class counts in train and val printed (roughly balanced).
- [ ] Sample grid image saved to `results/`.
- [ ] Gate: why we normalize pixels; why train/val/test are separate.

### Phase 1 — Binary logistic regression

Two classes (suggested: T-shirt/top vs Trouser). Agent implements sigmoid, binary cross-entropy, its gradient, and a batch gradient descent loop, tracking training loss and validation accuracy per epoch. The agent first derives the gradient step by step in plain language.

Done when:
- [ ] Loss decreases smoothly on the plot.
- [ ] Validation accuracy reported (should be very high on an easy pair; near 50% means a bug).
- [ ] Human has asked the agent to run 3+ learning rates and can describe what happens when it is too large or too small.
- [ ] Gate: what the gradient means intuitively; what the loss is measuring; why sigmoid.

### Phase 2 — Gradient checking

Agent writes `gradcheck.py`, which compares an analytic gradient with a numerical one (centered differences) and reports relative error. **The human runs it personally**, then asks the agent to introduce a deliberate bug and confirms the checker catches it.

Done when:
- [ ] Relative error on the correct implementation is small (around 1e-6 or lower).
- [ ] The deliberately buggy version fails the check.
- [ ] Gate: why numerical gradients are trustworthy but too slow for training; what a failing check tells you.

### Phase 3 — Softmax regression, 10 classes

Agent implements numerically stable softmax (subtract the row max), cross-entropy, its gradient, one-hot labels, and **mini-batch** SGD, plus a confusion matrix and per-class accuracy in `metrics.py`.

Done when:
- [ ] Gradient check passes (run by the human).
- [ ] Validation accuracy reported (roughly low-to-mid 80s percent is a sane range; far outside suggests a bug).
- [ ] Confusion matrix saved; human writes 2-3 sentences on which classes get confused and why.
- [ ] Gate: why the max-subtraction trick works; how softmax differs from sigmoid; what mini-batches trade off.

### Phase 4 — Two-layer neural network

Architecture: 784 → hidden (start at 128) → ReLU → 10 → softmax. Agent implements He initialization (scale by sqrt(2 / fan_in)), the forward pass with cached intermediates, and the backward pass via the chain rule. Agent first draws the computational graph in text and explains each gradient.

Done when:
- [ ] Gradient check passes on **both** weight matrices and both biases (run by the human).
- [ ] **Sanity check, run by the human:** the network overfits a tiny subset (~100 samples) to near 100% training accuracy. If not, there is a bug.
- [ ] Validation accuracy beats softmax regression.
- [ ] Gate: why initialization matters; what ReLU does and why a nonlinearity is needed; how a gradient flows backward through one layer.

### Phase 5 — Experiments and final evaluation

Agent writes an experiment runner that saves JSON and plots. Before each experiment the human states one line: "I expect X because Y." The agent then reports the real outcome, including when the human was wrong.

Suggested experiments:
- Learning rate sweep (log scale) for the MLP.
- Hidden size (32, 128, 512).
- Batch size (32, 128, 512).
- L2 regularization strength.
- Optional: momentum.

Then, **once**: choose the best config from validation results only and evaluate all three models on the test set.

Done when:
- [ ] Results table exists: model, key hyperparameters, validation accuracy, test accuracy.
- [ ] Every number came from an actual run.
- [ ] Gate: why the test set is used only once; what overfitting looks like in the curves; what the human would try next.

### Phase 6 — Write-up and tests

Agent writes:
- `pytest` tests: gradient checks per model, output shapes, softmax rows sum to 1, loss non-negative.
- `README.md`: problem, approach, how to run, results table, plots, and a short "what I learned" section drawn from the human's notes.

Done when:
- [ ] `pytest` passes from a fresh clone.
- [ ] README instructions work exactly as written.
- [ ] Final gate: the agent picks 5 random functions from `src/` and asks the human to explain what each does and why. Then a mock interview: 5 questions about the project, answered without looking at the code.

## Definition of done for Stage 1

- All three models implemented, gradient-checked, and evaluated on the held-out test set once.
- Clean repo, passing tests, honest README.
- The human can explain the math, the code, and the results without help.

## First message to send the agent

> Read PLAN.md fully and follow the roles and rules exactly. Start with Phase 0 only: explain what you're going to do, do it, walk me through the code, then run the understanding gate and stop so I can review before we continue.
