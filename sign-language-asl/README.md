# ASL Sign Language Recognition

Real-time American Sign Language hand gesture recognition using MediaPipe landmarks and a Random Forest classifier.

![Demo](demo.gif)

## Supported signs

| Category | Signs |
|---|---|
| Letters | A–Z (excluding J, Z — movement-based) |
| Words | YES, NO, HELLO, PLEASE, SORRY, THANK_YOU |

## Quick start

1. Clone the repo
2. `python -m venv venv && source venv/bin/activate`
3. `pip install -r requirements.txt`
4. Get Kaggle API token from kaggle.com → Account → Create New API Token
5. Place `kaggle.json` at `~/.kaggle/kaggle.json`
6. `python download_dataset.py`
7. `python train.py`
8. `python predict.py`

## Optional: add your own samples

If the model struggles to recognise your hand specifically:

```bash
python collect_data.py A  # repeat for any label you want to improve
python train.py           # retrain with combined data
```

## How it works

MediaPipe extracts 21 hand landmarks per frame → 63-float feature vector (x, y, z per landmark) → Random Forest classifier → 10-frame majority vote for stable predictions.

## Project structure

```
sign-language-asl/
├── .claude/
│   └── agents/
│       ├── orchestrator.md
│       ├── coder.md
│       └── reviewer.md
├── data/               # processed .npy landmark files
├── kaggle_data/        # raw downloaded Kaggle dataset
├── download_dataset.py # download + process Kaggle dataset
├── collect_data.py     # optional webcam data collection
├── train.py            # train Random Forest, save model.pkl
├── predict.py          # live webcam prediction
├── requirements.txt
├── CLAUDE.md
└── README.md
```

## Roadmap

- [ ] Dynamic gesture recognition with LSTM (full phrases)
- [ ] Streamlit web UI
- [ ] More word signs (WATER, FOOD, HELP, MORE)
- [ ] Model accuracy benchmarks table
