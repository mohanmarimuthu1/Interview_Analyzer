#!/usr/bin/env bash
# Downloads the public datasets used in M2-M4 into data/raw/.
# FER2013 needs the Kaggle CLI with a token in ~/.kaggle/kaggle.json and the
# competition rules accepted on kaggle.com first.
set -euo pipefail

RAW="${1:-data/raw}"
mkdir -p "$RAW"

fetch() {  # url dest
  if [ -f "$2" ]; then echo "skip $2 (exists)"; return; fi
  echo "get  $1"
  curl -fL --retry 3 -o "$2.part" "$1" && mv "$2.part" "$2" \
    || { rm -f "$2.part"; echo "FAILED $1 - download manually into $2"; }
}

# FER2013 (original csv, needed so FER+ labels line up row by row)
if command -v kaggle >/dev/null; then
  mkdir -p "$RAW/fer2013"
  kaggle competitions download \
    -c challenges-in-representation-learning-facial-expression-recognition-challenge \
    -p "$RAW/fer2013" || echo "FAILED fer2013 - accept the competition rules on kaggle.com"
else
  echo "kaggle CLI not found - skipping FER2013"
fi

# FER+ relabels
mkdir -p "$RAW/ferplus"
fetch https://raw.githubusercontent.com/microsoft/FERPlus/master/fer2013new.csv "$RAW/ferplus/fer2013new.csv"

# 300W-LP (head pose train) and AFLW2000-3D (head pose test), 3DDFA project page
mkdir -p "$RAW/headpose"
fetch http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/Database/AFLW2000-3D.zip "$RAW/headpose/AFLW2000-3D.zip"
fetch http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/Database/300W-LP/300W-LP.zip "$RAW/headpose/300W-LP.zip"

# MPIIFaceGaze
mkdir -p "$RAW/gaze"
fetch http://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIFaceGaze.zip "$RAW/gaze/MPIIFaceGaze.zip"

echo "done. unzip as needed; see data/README.md"
