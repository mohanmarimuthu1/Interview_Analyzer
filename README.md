# Interview Analyzer

A multimodal system that watches a candidate during a video interview and reads three signals at once — facial emotion, head pose, and eye gaze — to flag things a human interviewer might notice but can't easily quantify: nervousness, distraction, or lack of eye contact.

This is a Phase 1 prototype. The three perception models are trained and working individually; the fusion layer stitches their outputs together into one combined prediction per frame.

## What it actually does

For a given face image (or frame), the pipeline returns something like:

```python
{
  "Emotion": "neutral",
  "HeadPose": {"yaw": -4.2, "pitch": 1.8, "roll": 0.6},
  "EyeGaze": "center"
}
```

Three separate models run on the same input and their outputs get combined by `fusion_model.py`.

## The three models

**Facial Emotion Recognition** (`facial_emotion_recognition.py`)
ResNet18, fine-tuned on FER-2013. Classifies into 7 emotions — angry, disgust, fear, happy, neutral, sad, surprise.

**Head Pose Estimation** (`head_pose_model.py`)
ResNet50 (via `timm`) trained on the AFLW2000 dataset as a regression problem, predicting yaw/pitch/roll in degrees instead of discrete classes. Includes a fallback loader that auto-detects label format from CSV or TXT annotation files, since AFLW2000 dumps vary depending on where you get them.

**Eye Gaze Classification** (`eye_gaze_tracking.py`)
ViT-B/16 (`timm`, ImageNet-pretrained) fine-tuned on a Kaggle eye-gaze dataset to classify gaze direction.

**Fusion** (`fusion_model.py`)
Loads all three trained checkpoints and wraps them in a single `FusionModel` (`nn.Module`) that runs one forward pass and returns emotion logits, head-pose angles, and gaze logits together. Also has an evaluation path that reconstructs FER-2013 and eye-gaze test splits and reports per-model accuracy plus confusion matrices.

## Repo layout

```
eye_gaze_tracking.py           # ViT gaze classifier — training + inference
facial_emotion_recognition.py  # ResNet18 emotion classifier — training + eval
head_pose_model.py             # ResNet50 head-pose regressor — training + eval
fusion_model.py                # loads all 3 checkpoints, combines outputs
```

## Good to know before you run this

These files were exported straight out of Google Colab, so a few things come with that:

- They're notebooks in `.py` clothing — cells are laid out sequentially rather than wrapped in functions/`argparse`, and there are `!pip install` and `!kaggle datasets download` lines mixed into the code.
- Datasets are pulled from Kaggle at runtime (`kaggle.json` required), not bundled in the repo.
- Trained weights (`.pth` files) aren't checked in — training paths save to `/content/...` or Google Drive, which is Colab-specific.
- There's no single `main.py`, `requirements.txt`, or CLI entry point yet — each script is meant to be run top-to-bottom in Colab, or adapted cell-by-cell into a local script.

If you want to actually run this locally, you'll need to swap the Colab-specific paths (`/content/...`, `google.colab.drive`) for local ones and grab a `kaggle.json` API key.

## Datasets

| Model | Dataset |
|---|---|
| Emotion | [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013) |
| Head Pose | [AFLW2000 Head Pose Estimation](https://www.kaggle.com/datasets/androsstrk/aflw-2000-headpose-estimation-dataset) |
| Eye Gaze | [Eye Gaze Classification Dataset](https://www.kaggle.com/datasets/akshitgupta146/eye-gaze-classification-dataset) |

## Where this is headed

- Wrap the fusion output into a live webcam demo (there's a rough live-camera fusion block already, just needs cleanup)
- Package as an actual script/module instead of Colab exports
- Aggregate per-frame predictions into an interview-level summary (e.g. "distracted 20% of the time," "mostly neutral affect")
- Add proper requirements.txt and local dataset caching

## Author

Built by [Mohan Marimuthu](https://github.com/mohanmarimuthu1).
