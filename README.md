# Wayu-Paxa-TTS-Edge

Official inference for **Wayu-Paxa-TTS-Edge** — an 82M-parameter fixed-voice Thai–English
text-to-speech model: 12 voices, 24 kHz, Thai, English and code-switched text, fast enough
on CPU. Research artifact for the paper **Building and Evaluating Fixed-Voice Thai TTS from
Synthetic Speech**.

Not to be confused with **Paxa TTS Flash** — the two target different needs.
Wayu-Paxa-TTS-Edge is research-focused and small enough to run locally for personal use;
[Paxa TTS Flash](https://paxalabs.com/text-to-speech) is the production-ready service,
with markedly more realistic voices. If you are shipping to users, that is the one
you want.

> **Research artifact.** This repository accompanies the paper above and is released for
> reproducibility and further research. It is not a product: no maintenance or
> availability commitment and no warranty, and interfaces may change without notice
> between versions. There is no staffed support channel — but community help is very
> welcome, so please open an issue or a pull request.

| | |
|---|---|
| Model weights | [`wayu-ai/wayu-paxa-tts-edge`](https://huggingface.co/wayu-ai/wayu-paxa-tts-edge) — weights, voices and the model card |
| Evaluation | [`wayu-research/thai-tts-eval`](https://github.com/wayu-research/thai-tts-eval) — hard-keyword accuracy and pause placement |
| License | code Apache-2.0 · **weights [CC-BY-NC-4.0](https://creativecommons.org/licenses/by-nc/4.0/) — non-commercial** |

## Install

```bash
pip install git+https://github.com/wayu-research/wayu-tts-inference
apt-get install espeak-ng     # optional: English words no dictionary lists
```

## Run

```python
from wayu_tts import ThaiTTS

tts = ThaiTTS.from_pretrained("wayu-ai/wayu-paxa-tts-edge")
audio = tts("เมื่อวานเรียก Grab ไปทำงาน รถมาเร็วมาก", voice="m_young_clear")
tts.save("out.wav", audio)
```

```bash
wayu-tts "สวัสดีครับ วันนี้อากาศดีมาก" --voice m_young_clear --out hello.wav
wayu-tts --list-voices
wayu-tts "ราคา 1,250 บาท" --phonemes      # what the model will actually be fed
wayu-tts "ประโยคเดิม" --seed 0            # reproducible generation
```

Text normalization (numbers, ฿, %, ๆ, abbreviations), Thai–English script routing, and
sentence splitting for long text are all handled internally — hand it written Thai and
it speaks. `audio` is float32 numpy at `tts.sample_rate` (24 kHz). Pass `seed=` when
generation must be reproducible.

## Voices

Twelve voices, distilled from a zero-shot voice-cloning teacher (OmniVoice) into fixed
speakers. Because OmniVoice was trained on in-the-wild speech, these synthetic voices may
incidentally resemble real individuals; we did not crawl speech data or intentionally clone
any real person.

| voice | design | delivery |
|---|---|---|
| `f_teen_bright` | female, teenager, high pitch | engaging |
| `f_young_bright` | female, young adult, high pitch | engaging |
| `f_young_clear` | female, young adult, moderate pitch | neutral |
| `f_young_warm` | female, young adult, low pitch | read |
| `f_mid_clear` | female, middle-aged, moderate pitch | neutral |
| `f_mid_warm` | female, middle-aged, low pitch | read |
| `f_elderly_soft` | female, elderly, moderate pitch | neutral |
| `f_elderly_low` | female, elderly, low pitch | read |
| `m_teen_bright` | male, teenager, high pitch | engaging |
| `m_young_clear` | male, young adult, moderate pitch | neutral |
| `m_mid_warm` | male, middle-aged, low pitch | read |
| `m_elderly_deep` | male, elderly, very low pitch | read |

Each voice ships with a calibrated speaking rate (`voice_speeds` in `config.json`),
applied automatically. `speed=` multiplies it: 1.0 is "this voice at its intended pace",
`>1` is faster.

## Model directory

`from_pretrained` accepts a Hugging Face repo id or a local directory:

```
config.json        architecture + vocabulary + the Thai token contract
model.pth          bert / bert_encoder / predictor / text_encoder / decoder
voices/*.pt        one style vector per sentence length, per speaker
```

The weights also load with the upstream model class directly, but that path skips this
package's text frontend and rate calibration — use it only if you bring your own.

## Development

```bash
pip install -e ".[dev]"
pytest          # frontend tests; no model download
ruff check .
```

## Limitations

- **Central Thai only.** No dialect coverage.
- **The G2P is dictionary-based** (tltk). It mis-segments some informal spellings and
  compounds, occasionally dropping a syllable without an error. When the output has to
  be right, inspect `--phonemes` (or `phonemize()`) and reword until it reads clean.
- **Long sentences drift** in speaking rate past the lengths seen in training.
- **Fixed voices.** No voice cloning path, no reference-audio input.
- **No training code.** This repo serves a checkpoint; it does not build one.
- **Open-source reproduction of the paper's frontend.** This package re-implements the
  text frontend described in the paper; it is not the code the paper's numbers were
  produced with, and any part of the implementation may differ, so benchmark scores can
  differ from the paper's.

## Disclaimer

Provided **"as is", without warranty of any kind**, express or implied, to the fullest
extent permitted by law — see sections 7 and 8 of [`LICENSE`](LICENSE) for the code, and
section 5 of CC-BY-NC-4.0 for the weights. The authors, the maintainers and their
affiliated institutions accept **no responsibility and no liability** for any damage,
loss, cost or claim arising from use or misuse of this software, the model weights, or
any audio produced with them, and are not responsible for how third parties use them.

This model generates **synthetic speech**. You are solely responsible for what you
synthesize and for complying with the law wherever you deploy it. In particular: do not
present its output as a recording of a real person, do not use it to impersonate anyone
or to produce misleading, defamatory or fraudulent audio, and disclose that audio is
machine-generated wherever a listener could reasonably be misled.

## License and provenance

**The code in this repository is Apache-2.0. The model weights are not.** They are
released under [CC-BY-NC-4.0](https://creativecommons.org/licenses/by-nc/4.0/):
attribution required, **commercial use not permitted**. If you need Thai TTS in a
commercial product, use [Paxa TTS Flash](https://paxalabs.com/text-to-speech).

The weights are a fine-tune of [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)
(Apache-2.0), built on [StyleTTS2](https://github.com/yl4579/StyleTTS2) (MIT). That
upstream license and its attribution are retained: the non-commercial term covers this
fine-tune only and takes nothing away from the base model, which stays available under
its own terms. Voices are synthetic, distilled from a voice-cloning teacher on designed
reference prompts; see Voices above for the note on incidental resemblance to real
individuals.

**Acceptable use.** By downloading or using this model you agree to the Wayu Research
[Acceptable Use terms](https://www.wayuresearch.org/terms#acceptable-use).

## Citation

```bibtex
@misc{pipatanakul2026buildingevaluatingfixedvoicethai,
      title={Building and Evaluating Fixed-Voice Thai TTS from Synthetic Speech}, 
      author={Kunat Pipatanakul and Potsawee Manakul and Warit Sirichotedumrong and Sittipong Sripaisarnmongkol and Pakorn Nathong and Phatrasek Jirabovonvisut},
      year={2026},
      eprint={2609.03502},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2609.03502}, 
}
```