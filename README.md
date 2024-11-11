# Min-S2ST

**Min-S2ST** (Minimal Speech-to-Speech Translation) is a Python-based speech-to-speech translation pipeline that uses pre-trained models from Hugging Face to transcribe, translate, and synthesize audio from a source language to a target language.

## Features

- **Automatic Speech Recognition (ASR)**: Detects the language and transcribes speech from an audio file.
- **Translation**: Translates the transcribed text into the target language.
- **Text-to-Speech (TTS)**: Synthesizes the translated text into speech in the target language.

## Installation

Install the required dependencies:

```bash
pip install transformers espnet torchaudio pydub soundfile
```

## Usage

1. Place the source audio file in the root directory (e.g., `./accidents.wav`).
2. Run the script to perform the speech-to-speech translation:

    ```python
    python main.py path/to/input.wav path/to/output.wav
    ```

The output will be saved as `result.wav`.

## Code Structure

- **`initialize_models()`**: Loads ASR, Translation, and TTS models.
- **`transcribe_audio()`**: Transcribes audio into text with timestamps.
- **`translate_segments()`**: Translates each transcribed segment to the target language.
- **`synthesize_and_merge_segments()`**: Synthesizes and merges the translated segments into a single output audio file.
- **`process_audio()`**: The main function that runs the end-to-end pipeline.

## Example

For an input audio file in English, `Min-S2ST` can automatically detect the language, translate it to Spanish, and output a Spanish audio file.

## License

This project is licensed under the MIT License.