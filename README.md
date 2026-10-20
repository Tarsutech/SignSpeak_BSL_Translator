# SignSpeak

A two-way British Sign Language fingerspelling translator that runs on a laptop webcam.

- **Sign to speech**: fingerspell in front of the camera, SignSpeak types the letters and reads the words out loud.
- **Speech to sign**: say something, SignSpeak spells it back letter by letter as BSL fingerspelling cards.

Everything runs in one window and nothing freezes while it listens, speaks or spells.

## Setup

Needs Python 3.9 to 3.12 and a webcam.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Voice input uses Google's free speech service, so it needs an internet connection. If PyAudio won't install, run `python main.py --no-voice` and everything else still works.

## First run: teach it your signing

BSL fingerspelling is two-handed and everyone signs a little differently, so SignSpeak learns from you. It takes about five minutes for the whole alphabet.

1. Press **T** to open training.
2. Make the sign for a letter and hold it.
3. Press that letter's key. It records 40 snapshots in about two seconds. Move your hands slightly while it records so it learns small variations.
4. Repeat for each letter, then press **Enter** to go back to translating.

Training is saved to `data/model.npz` and loads automatically next time. To redo a letter, press **Backspace** then the letter in training mode, and record it again.

Tips: train in the lighting you'll use, keep both hands fully in frame, and add a second round of samples for letters it mixes up.

## Controls

| Translate mode | |
|---|---|
| Space | add a space (one is also added after 1.5 s with no hands) |
| Backspace | delete the last letter |
| C | speak the signed text |
| V | listen, then fingerspell what you said |
| R | clear everything |
| T | training mode |
| L | show or hide hand landmarks |
| H | show or hide the key guide |
| Q or Esc | quit |

To sign a double letter (the LL in HELLO), drop or change your hands for a moment between the two.

## Letter pictures

The speech-to-sign panel shows a card per letter. Put your own pictures in `assets/bsl_letters/` named `A.png` to `Z.png`. Letters without a picture show a plain letter card.

## Options

```
python main.py --camera 1          use a different webcam
python main.py --accurate          heavier hand model, more accurate, slower
python main.py --width 960 --height 540   lower resolution for slower laptops
python main.py --no-voice          no microphone or speech
```

Other settings (how long a letter must be held, confidence threshold and so on) are in `signspeak/config.py`.

## How it works

```
camera thread ──> MediaPipe hands ──> features ──> kNN classifier ──> stabiliser ──> text ──> speech
microphone thread ──> speech to text ──> letters ──> fingerspelling player ──> letter cards
```

- **Features** (`recognition/features.py`): each hand's 21 landmarks are centred on the wrist and scaled by hand size, so it doesn't matter where you sit or how close you are. Both hands are included, plus where they are relative to each other, because most BSL letters touch one hand with the other.
- **Classifier** (`recognition/classifier.py`): k-nearest neighbours on your own samples. It trains instantly and redoing one letter doesn't affect the rest. Poses that don't look like anything trained are ignored.
- **Stabiliser** (`recognition/stabiliser.py`): a letter is typed only after it has been held steadily for 8 frames, and only once per hold.
- **Threads**: the camera, speech output and speech input each run in the background, so the video never stalls.

## Limits

- Only static handshapes. BSL **H** and **J** involve movement and may be less reliable.
- It recognises fingerspelling, not full BSL signs or grammar.
- Trained on one person, it works best for that person.

## Project layout

```
main.py                     start here
signspeak/
  app.py                    window and main loop
  controller.py             modes, keys, recognition, training
  config.py                 all settings
  text.py                   text helpers
  vision/camera.py          threaded webcam
  vision/hands.py           MediaPipe wrapper
  recognition/              features, classifier, stabiliser
  audio/tts.py, stt.py      speech out and in
  ui/hud.py                 on-screen display
  ui/fingerspell.py         letter cards and player
tests/test_core.py          python -m pytest
```
