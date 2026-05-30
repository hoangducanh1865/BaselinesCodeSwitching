import numpy as np
import os

import torch
import torchaudio.transforms as at
import torchaudio
import editdistance
import av
import pandas as pd
import json
import random

class calc_metrics:
    def __init__(self):
        pass
    def __call__(self, refs, preds):
        """
        refs are output from dataloader, so uses the collate fn, that already contains the normalization
        preds are the output of whisper tokenizer, which doesn't have dataset specific normalization

        they should both in list (list of list)
        """
        distance = 0
        tokens = 0
        wer_list = []
        processed_preds = []
        processed_refs = []
        exclude = [",", "?", ".", "!", ";"]
        for ref, pred in zip(refs, preds):
            pred = pred.lower()
            pred = ''.join(ch for ch in pred if ch not in exclude)
            processed_preds.append(pred)
            processed_refs.append(ref) # do not process ref
            cur_dist =editdistance.distance(pred.split(" "), ref.split(" "))
            cur_tokens = len(ref.split(" "))
            wer_list.append(cur_dist/cur_tokens)
            distance += cur_dist
            tokens += cur_tokens

        return {"wer":distance/tokens}, (wer_list, processed_preds, processed_refs)


def load_wave(wave_path, sample_rate:int=16000) -> torch.Tensor:
    with av.open(wave_path, metadata_errors="ignore") as container:
        decode = container.decode(audio=0)
        aframes_list = [frame.to_ndarray() for frame in decode]
        aframes = np.concatenate(aframes_list, 1)
        # Convert to float32 for processing. We normalize by dividing by 32768.0 (2^15) to get range [-1, 1]
        wav = torch.from_numpy(aframes).float() / 32768.0
        wav = wav.mean(dim=0)  # Taking the mean to convert from stereo to mono
        cur_sample_rate = container.streams.audio[0].rate
        if cur_sample_rate != sample_rate:
            resampler = at.Resample(orig_freq=cur_sample_rate, new_freq=sample_rate)
            wav = resampler(wav)
        if wav.mean() == 0:
            print(wave_path, "empty!")
    return wav

class CodeMixedWhisperDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, dataset_name, feature_extractor, tokenizer, sample_rate=16000, prompt=True, max_prompt_length=190, max_seq_length=448, train=True, test=False):
        """
        A dataset class for code-mixed speech recognition tasks with prompts.

        Args:
            csv_path (str): Path to the CSV file containing dataset information.
            feature_extractor (Any): Feature extractor for processing audio files.
            tokenizer (Any): Tokenizer for encoding text and prompts.
            sample_rate (int): Sampling rate for audio processing. Defaults to 16000.
            prompt (bool): Whether to include prompts in the output. Defaults to True.
            max_prompt_length (int): Maximum length for the prompt after truncation. Defaults to 190.
        """
        super().__init__()
        self.data = pd.read_csv(csv_path)
        if test:
            self.data = pd.read_csv(csv_path)

        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.prompt = prompt
        self.max_prompt_length = max_prompt_length
        self.max_seq_length = max_seq_length
        self.dataset_name = dataset_name

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves a single data sample.

        Args:
            idx (int): Index of the data sample.

        Returns:
            dict: A dictionary containing processed audio, prompt, and labels.
        """
        # Load data from the CSV
        audio_path = self.data.iloc[idx]["path"]
        text = self.data.iloc[idx]["text"]
        # prompt_text = """
        # This transcript is a code-switched speech. language labels of words i.e (hindi (hi) and english(en)) in the transcripts are {}.
        # Use this to transcribe the audio.
        # """.format(self.data.iloc[idx]['language_labels'])
        # Text is related to tutorials on academic or technical subjects.
        
        # prompt_text = """
        # This transcript is a code-switched text. Mix of devnagri and english words are present. 
        # Text is related to tutorials on academic or technical subjects.
        # """

        prompt_text = """
        This transcript is a code-switched text. Mix of devnagri and english words are present. 
        # Text is related to tutorials on academic or technical subjects.
        Few examples look like this
        1. तो electricity bill option पर click करें
        2. कुछ गलत password दीजिए और enter प्रेस करें
        """

        # prompt_text = """
        # This transcript is a code-switched text. Mix of devnagri and english words are present. 
        # Text is related to tutorials on academic or technical subjects.
        # one example looks like this: तो electricity bill option पर click करें
        # decode in this format.
        # """

        # prompt_text = """
        # This transcript is a code-switched text from the famous TV show 'Sarabhai vs Sarabhai'.
        # Mix of devanagri and english words are present but all words are written in romanized english text.
        # """

        # prompt_text = """
        # The transcript comprises of telephone quality speech data in Hindi. Transcript is a code-switched text but script is devnagri only.
        # Transcribe the audio accordingly.
        # """

        # Load and process audio
        # print(f'audio path is this: {audio_path}')
        # print(f'audio text is this: {text}')
        audio, sr = torchaudio.load(audio_path)
        
        # if self.dataset_name == 'gramvani':
        #     #resample te audio to 16KHz
        #     resampler = at.Resample(orig_freq=sr, new_freq=self.sample_rate)
        #     audio = resampler(audio)
        
        audio = audio.squeeze().numpy()
        processed_audio = self.feature_extractor(audio, sampling_rate=self.sample_rate).input_features
        processed_audio = torch.tensor(processed_audio[0])  # Convert to PyTorch tensor

        # Encode text (labels)
        encoded_labels = torch.tensor(self.tokenizer.encode(text.lower(), add_special_tokens=True))

        if self.prompt:
            # Encode the prompt
            encoded_prompt = self.tokenizer.encode(prompt_text.lower(), add_special_tokens=False)

            # Truncate the prompt if necessary
            if len(encoded_prompt) > self.max_prompt_length:
                encoded_prompt = encoded_prompt[:self.max_prompt_length]

            encoded_prompt = torch.tensor(encoded_prompt)  # Convert to PyTorch tensor

            return {
                "input_features": processed_audio,
                "prompt": encoded_prompt,
                "labels": encoded_labels
            }
        else:
            # If prompt is not included, raise an error
            raise ValueError("Prompts must be enabled for this dataset.")
        

class CodeMixedWhisperDataset_V2(torch.utils.data.Dataset):
    def __init__(self, csv_path, feature_extractor, tokenizer, sample_rate=16000):
        """
        A dataset class for code-mixed speech recognition tasks with prompts.

        Args:
            csv_path (str): Path to the CSV file containing dataset information.
            feature_extractor (Any): Feature extractor for processing audio files.
            tokenizer (Any): Tokenizer for encoding text and prompts.
            sample_rate (int): Sampling rate for audio processing. Defaults to 16000.
            prompt (bool): Whether to include prompts in the output. Defaults to True.
            max_prompt_length (int): Maximum length for the prompt after truncation. Defaults to 190.
        """
        super().__init__()
        self.data = pd.read_csv(csv_path)

        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves a single data sample.

        Args:
            idx (int): Index of the data sample.

        Returns:
            dict: A dictionary containing processed audio, prompt, and labels.
        """
        # Load data from the CSV
        audio_path = self.data.iloc[idx]["path"]
        text = self.data.iloc[idx]["text"]

        # Load and process audio
        audio, _ = torchaudio.load(audio_path)
        audio = audio.squeeze().numpy()
        processed_audio = self.feature_extractor(audio, sampling_rate=self.sample_rate).input_features
        processed_audio = torch.tensor(processed_audio[0])  # Convert to PyTorch tensor

        # Encode text (labels)
        encoded_labels = torch.tensor(self.tokenizer.encode(text.lower(), add_special_tokens=True))

        return {
            "input_features": processed_audio,
            "labels": encoded_labels
        }


class PromptWhisperDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, phase, feature_extractor, tokenizer, prompt=False, audio_type=".wav", sample_rate=16000, random=False, basic=False):
        super().__init__()
        self.phase = phase
        self.base_path = base_path
        self.sample_rate = sample_rate
        self.prompt = prompt
        self.random_prompt = random
        self.data = []
        self.prompt_pool = []
        self.audio_type = audio_type
        self.basic = basic
        self._load_data()
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer
        
    def _initialize_prompt_pool(self):
        # Walk through the directory structure to build the prompt pool
        for root, dirs, files in os.walk(os.path.join(self.base_path, self.phase)):
            json_files = [f for f in files if f.endswith('.json')]
            for json_file_name in json_files:
                json_file_path = os.path.join(root, json_file_name)
                with open(json_file_path, 'r', encoding='utf-8') as json_file:
                    json_data = json.load(json_file)
                    prompt = json_data.get("prompt1_response", "")
                    if prompt:
                        self.prompt_pool.append(prompt)

                    
    def _load_data(self):
        # Walk through the directory structure
        for root, dirs, files in os.walk(os.path.join(self.base_path, self.phase)):
            wav_files = [f for f in files if f.endswith(f'{self.audio_type}')]
            json_files = [f for f in files if f.endswith('.json')]
            for wav_file in wav_files:
                base_name = os.path.splitext(wav_file)[0]
                json_file_name = f"{base_name}.json"
                if json_file_name in json_files:
                    json_file_path = os.path.join(root, json_file_name)
                    # Open the json file and extract "text" and "prompt"
                    with open(json_file_path, 'r', encoding='utf-8') as json_file:
                        json_data = json.load(json_file)
                        text = json_data.get("text", "")
                        prompt = json_data.get("prompt1_response", "")
                        random_prompt = random.choice(self.prompt_pool) if self.prompt_pool else ""
                        basic = json_data.get("basic", "")
                    self.data.append([os.path.join(root, wav_file),
                        prompt,
                        random_prompt,
                        basic,
                        text
                    ])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, id):
        audio_path, prompt, random_prompt, basic_prompt, raw_text = self.data[id]
        # Load and process audio
        audio, _ = torchaudio.load(audio_path)
        audio = audio.squeeze().numpy()  # Converting to NumPy array if not already
        processed_audio = self.feature_extractor(audio, sampling_rate=self.sample_rate).input_features
        processed_audio = torch.tensor(processed_audio[0])  # Ensure processed_audio is a tensor
        # Encode text
        encoded_labels = torch.tensor(self.tokenizer.encode(raw_text.lower()))  # Convert to tensor
        
        if self.prompt:
            if self.random_prompt and 'train' in self.phase:
                if torch.rand([]) < 0.05 and 'train' in self.phase:
                    encoded_prompt = self.tokenizer.encode(random_prompt.lower(), add_special_tokens=False)
                else:
                    encoded_prompt = self.tokenizer.encode(prompt.lower(), add_special_tokens=False)
            elif self.basic:
                encoded_prompt = self.tokenizer.encode(basic_prompt.lower(), add_special_tokens=False)
            else:
                encoded_prompt = self.tokenizer.encode(prompt.lower(), add_special_tokens=False)
                
            if len(encoded_prompt) > 190:
                encoded_prompt = encoded_prompt[:190]
            
            encoded_prompt = torch.tensor(encoded_prompt)  # Ensure encoded_prompt is a tensor
            return {
                "input_features": processed_audio,
                "prompt": encoded_prompt,  # Including the prompt in the output
                "labels": encoded_labels
            }
        else:
            print("prompt must be used.")
            raise(ValueError)