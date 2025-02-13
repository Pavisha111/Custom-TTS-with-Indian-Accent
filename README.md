# Custom-TTS-with-Indian-Accent


ls checkpoints_v2/converter/
!wget https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip
!unzip checkpoints_v2_0417.zip
tone_color_converter = ToneColorConverter(f'path_to_correct_directory/config.json', device=device)

import os
import torch
from openvoice import se_extractor
from openvoice.api import ToneColorConverter
from melo.api import TTS
import re
from IPython.display import Audio
device = "cuda:0" if torch.cuda.is_available() else "cpu"
output_dir = 'outputs_v2'
ckpt_converter = 'checkpoints_v2/converter'
tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')
os.makedirs(output_dir, exist_ok=True)
Audio("/home/bbbs/Documents/pdf/Riya.mp3")
reference_speaker = "/home/bbbs/Documents/pdf/Riya.mp3"
target_se, audio_name = se_extractor.get_se(reference_speaker, tone_color_converter, vad=True)

source_se = torch.load(f'checkpoints_v2/base_speakers/ses/en-india.pth', map_location=device)



class TextProcessor:
    def __init__(self, device, speaker_key="EN_INDIA"):
        self.device = device
        self.speaker_key = speaker_key

    def handle_acronyms(self, text):
        # Define specific acronyms and their replacements
        replacements = {
            "HZ": "hertz",
            "TXC": " T-X-C",
            "LCP": "L-C-P",
            "RX": "R-X",
            "TX": " T -X",
            "TWT": " T -W-T",
            "RDY": "ready",
            "RPM": "R-P-M",
            "STND BY": "standby",
            "TRMT": "transmit"
        }

        # Sort by length to ensure multi-word acronyms are replaced first
        for key, value in sorted(replacements.items(), key=lambda x: -len(x[0])):
            text = re.sub(rf'\b{re.escape(key)}\b', value, text)

        # Return the processed text
        return text

    def handle_symbols(self, text):
        # Define symbol replacements
        symbol_replacements = {
            r'/': ' by ',
            r'\\': ' backslash ',
            r'-': '         ',
            r'_': ' underscore ',
            r';': ' semicolon ',
            r':': ' colon ',
            r'@': ' at ',
            r'#': ' hash ',
            r'&': ' and ',
            r'?': '        ',
            r'!': ' exclamation mark ',
            r'"': ' quote ',
            r"'": ' single quote ',
            r'(': '           ',
            r')': '            ',
        }

        # Replace decimals (e.g., "3.5" -> "3 point 5")
        text = re.sub(r'(\d)\.(\d)', r'\1 point \2', text)

        # Apply symbol replacements
        for symbol, replacement in symbol_replacements.items():
            text = text.replace(symbol, replacement)

        return text

    def handle_sentence_endings(self, text):
        # Add spaces after periods and commas to simulate natural pauses
        text = re.sub(r'(\.)', r'\1  ', text)
        text = re.sub(r'(,)', r'\1  ', text)
        return text

    def process_text(self, text):
        # Process acronyms, symbols, and sentence endings
        text = self.handle_acronyms(text)
        text = self.handle_symbols(text)
        text = self.handle_sentence_endings(text)
        return text

    def tts(self, text):
        # Process the input text
        processed_text = self.process_text(text)

        # Initialize TTS model
        model = TTS(language='EN', device=self.device)
        src_path = 'tmp.wav'
        speaker_ids = model.hps.data.spk2id

        try:
            # Try getting the speaker ID
            speaker_id = speaker_ids[self.speaker_key]
        except KeyError:
            raise ValueError(f"Speaker key '{self.speaker_key}' not found in speaker_ids.")

        print(f"Processed text: {processed_text}")
        # Generate the speech file
        model.tts_to_file(processed_text, speaker_id, src_path, speed=0.7)
        print("Audio saved to output.wav")

input_text = "Note: To avoid damage to the TWT, it is recommended not to stay longer than 8 hours successively in the STND BY mode. After this time the high voltage has to be switched on for about 15 minutes."
save_path = '6.wav'

processor = TextProcessor(device)
processor.tts(input_text)
encode_message = "@MyShell"
tone_color_converter.convert(
    audio_src_path='tmp.wav',
    src_se=source_se,
    tgt_se=target_se,
    output_path=save_path,
    message=encode_message
)
Audio(save_path)



