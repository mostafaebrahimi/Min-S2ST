from transformers import pipeline
from espnet2.bin.tts_inference import Text2Speech
import soundfile as sf
import numpy as np
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Min-S2ST Speech-to-Speech-Translation")
    parser.add_argument("input_file", type=str, help="Path to the input audio file")
    parser.add_argument("output_file", type=str, help="Path to the output audio file")
    args = parser.parse_args()
    return args.input_file, args.output_file


def initialize_models():
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-base")
    translation = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")
    tts = Text2Speech.from_pretrained("espnet/kan-bayashi_ljspeech_vits")
    return asr, translation, tts

def transcribe_audio(asr, audio_path):
    transcription = asr(audio_path, return_timestamps=True)
    if "chunks" in transcription:
        return [
            {
                "start": chunk["timestamp"][0],
                "end": chunk["timestamp"][1],
                "text": chunk["text"]
            }
            for chunk in transcription["chunks"]
        ]
    else:
        return [{"start": 0, "end": None, "text": transcription["text"]}]

def translate_segments(translation, segments):
    return [
        {
            "start": seg["start"],
            "end": seg["end"],
            "text": translation(seg["text"])[0]["translation_text"]
        }
        for seg in segments
    ]

def synthesize_and_merge_segments(tts, translated_segments, output_path, sample_rate=22050):
    final_audio = []
    current_time = 0

    for seg in translated_segments:
        if seg["start"] is None or seg["end"] is None:
            print(f"Skipping segment due to missing timestamps: {seg}")
            continue

        try:
            tts_output = tts(seg["text"])
            waveform = tts_output["wav"] if isinstance(tts_output, dict) else tts_output

            segment_start_time_samples = int(seg["start"] * sample_rate)
            silence_duration = max(0, segment_start_time_samples - len(final_audio))
            final_audio.extend(np.zeros(silence_duration, dtype=np.float32))
            final_audio.extend(waveform.numpy())

        except Exception as e:
            print(f"Error generating audio for segment '{seg['text']}': {e}")
            continue

    sf.write(output_path, np.array(final_audio), sample_rate)
    print(f"Translated audio saved as {output_path}")

def process_audio(input_path, output_path):
    asr, translation, tts = initialize_models()
    segments = transcribe_audio(asr, input_path)
    print("Transcription Complete.")
    translated_segments = translate_segments(translation, segments)
    print("Translation Complete.")
    synthesize_and_merge_segments(tts, translated_segments, output_path)
    print("Synthesis and merging complete.")

input_path, output_path = parse_arguments()
process_audio(input_path, output_path)
