# Ethics and intended use

## Intended use
InterviewLens is a practice tool for the candidate. You record yourself
answering interview questions and get feedback on your own delivery: gaze,
head movement, expression, speaking rate, filler words, pauses. It runs
locally and nothing is uploaded unless you turn that on.

## Not intended for
- Screening, ranking or rejecting job candidates
- Any decision about a person made by an employer, school or other third party
- Inferring emotions of employees or students. Under the EU AI Act,
  Art. 5(1)(f), emotion recognition in workplace and education settings is
  prohibited (applies since February 2025).

## How output is presented
Scores always come with the behaviour behind them, for example "gaze off
camera 38% of the answer" or "speech rate 190 wpm". There is no bare
"confidence" label without evidence.

## Known limitations and biases
- Emotion, gaze and head pose models are trained on public datasets that are
  skewed in age, ethnicity and recording conditions.
- First Impressions V2 is YouTube vlogs, not real interviews, and its labels
  are crowd impressions, which carry the raters' biases.
- Per-group error is measured and reported (milestone M7). Read
  `reports/fairness.md` before trusting any score.

## Data
See `data/LICENSES.md`. Most datasets here are non-commercial research only.
