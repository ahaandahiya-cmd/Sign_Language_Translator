# ASL Fingerspelling → Text

Real-time webcam app that reads ASL fingerspelling (the A-Z hand alphabet)
and types it out as text. Runs fully locally — nothing leaves your machine.

## Setup

```bash
pip install -r requirements.txt
python main.py
```

That's it — it works immediately using a built-in rule-based classifier
(geometric hand-shape rules, no training needed).

**First run only:** it auto-downloads the hand-tracking model (~10MB) from
Google's servers into the project folder, so you'll need an internet
connection the very first time. After that it runs fully offline.

Uses MediaPipe's current Tasks API under the hood (older `mediapipe.solutions`
was removed in recent mediapipe releases, so this avoids that entirely).

## Getting better accuracy (recommended)

The rule-based classifier is a reasonable starting point, but several ASL
letters are genuinely hard to tell apart from geometry alone — they differ
by hand *orientation* (P vs K, Q vs G) or by subtle thumb placement
(M vs N vs S vs T vs E). Rules alone hit a ceiling here.

The fix: calibrate on your own hand.

```bash
python collect_data.py   # hold each letter shape, press its key to record samples
python train_model.py    # trains a KNN model on your samples
python main.py           # now automatically uses your trained model
```

Aim for 10-20 samples per letter, from a couple of slightly different
hand angles/distances each. `main.py` prefers `model.pkl` automatically if
it exists — delete it to fall back to the rule-based classifier.

## Controls (in the main app)

| Key | Action |
|---|---|
| `SPACE` | insert a space |
| `BACKSPACE` | delete last character |
| `C` | clear the line |
| `S` | append current line to `transcript.txt` |
| `Q` / `ESC` | quit |

A letter only gets typed after you hold the shape steady for a moment
(~0.4s), so it doesn't spam the same letter every frame.

## Scope & limitations

- **Static letters only.** `J` and `Z` involve motion (a drawn trace) and
  aren't supported — this reads hand *shape*, not motion, in v1.
- **Rule-based classifier covers:** A, B, C, D, E, F, I, L, M, N, O, S, T,
  U, V, W, X, Y reliably. G, H, K, P, Q, R are present in the code but
  low-confidence without calibration — train on your own hand for these.
- **Lighting/background** matter — MediaPipe's hand detection does best
  with decent lighting and your hand clearly separated from the background.
- This is fingerspelling recognition, not full ASL — full ASL is a
  distinct language with its own grammar, facial grammar, and thousands of
  non-alphabetic signs; that's a much larger project than an alphabet reader.

## Project layout

```
main.py             — the real-time app (run this)
mp_hand_detector.py   — MediaPipe Tasks API wrapper + landmark drawing
hand_features.py       — landmark geometry shared by both classifiers
rule_classifier.py      — no-training heuristic classifier
collect_data.py          — records labeled samples of your hand -> data/asl_landmarks.csv
train_model.py            — trains model.pkl from collected samples
requirements.txt
```

## Natural next steps

- Add `J`/`Z` via short motion-sequence tracking (buffer of landmark
  positions over ~1s, matched against the letter's known trace shape).
- Swap KNN for a small neural net if you collect a larger dataset — same
  `normalized_vector` features would feed it directly.
- Word-level signs (not just fingerspelling) would need a labeled video
  dataset per sign and a sequence model (e.g. an LSTM/temporal CNN over
  landmark sequences) — a meaningfully bigger project than this one.
