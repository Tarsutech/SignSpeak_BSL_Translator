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

For the microphone (voice to sign), also install PyAudio. On a Mac run `brew install portaudio` first, then:

```bash
pip install -r requirements-voice.txt
```

Without it, everything else still works; the V key just says voice input isn't available. Voice input uses Google's free speech service, so it needs an internet connection.

## First run: teach it your signing

BSL fingerspelling is two-handed and everyone signs a little differently, so SignSpeak learns from you. It takes about five minutes for the whole alphabet.

1. Press **T** to open training.
2. Make the sign for a letter and hold it.
3. Press that letter's key. It records 40 snapshots in about two seconds. Move your hands slightly while it records so it learns small variations.
4. Repeat for each letter, then press **Enter** to go back to translating.

Training is saved to `data/model.npz` and loads automatically next time. To redo a letter, press **Backspace** then the letter in training mode, and record it again.

### Getting it accurate

- **Record each letter 2–3 times**, not once. Press the key again to add another round. Shift your hands a little between rounds (closer, further, slightly tilted). Variety matters more than quantity.
- Keep **both hands fully in frame** and use the lighting you'll actually sign in.
- If it shows **"or X?"** under the guess, it's torn between two letters. Add another round of both.
- Is a letter just wrong? Press **Backspace then the letter** to forget it, and record it fresh.

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
python main.py --fast              lighter hand model for slow laptops (less accurate)
python main.py --width 960 --height 540   lower resolution for slower laptops
python main.py --no-voice          no microphone or speech
```

Other settings (how long a letter must be held, confidence threshold and so on) are in `signspeak/config.py`.

## How it works

```
camera thread ──> MediaPipe hands ──> features ──> kNN classifier ──> stabiliser ──> text ──> speech
microphone thread ──> speech to text ──> letters ──> fingerspelling player ──> letter cards
```

- **Features** (`recognition/features.py`): each hand's shape is centred on the wrist and scaled by palm size, so it doesn't matter where you sit or how close you are. Hands are ordered by where they are on screen, not MediaPipe's left/right labels, which flip when hands touch. On top of that it measures the distance from every fingertip of one hand to every fingertip of the other, because in BSL *which finger touches where* is what tells most letters apart.
- **Training** stores raw landmarks, then adds slightly rotated and jittered copies of each frame, so 40 recorded frames act like 160 and cope with small differences on another day.
- **Classifier** (`recognition/classifier.py`): k-nearest neighbours over your examples. It trains instantly, takes under 1 ms per frame, and redoing one letter doesn't affect the rest.
- **Smoothing and voting**: the last 4 frames of landmarks are averaged to calm tracking jitter, then the last 10 guesses vote. A letter is typed once it gets 7 votes, so the odd wrong frame doesn't reset it.
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
