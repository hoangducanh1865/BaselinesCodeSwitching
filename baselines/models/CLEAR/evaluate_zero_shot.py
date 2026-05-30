from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torchaudio
import pandas as pd
import random
import tqdm
import torch   


processor = WhisperProcessor.from_pretrained('openai/whisper-small', language='hi', task='transcribe')
model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-small').to('cuda')
 

df = pd.read_csv('/raid/home/shada/Improving-ASR-with-LLM-Description/data/blind_data_mod.csv')

def generate_predictions(df):
    pre = []
    ref = []
    for idx in tqdm.tqdm(range(len(df))):
        audio, sr = torchaudio.load(df.iloc[idx]['path'])
        input_ids = processor(audio[0], sampling_rate=16000, return_tensors='pt').input_features.to('cuda')
        with torch.no_grad():
            predicted_ids = model.generate(input_ids)
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
            ref.append(df.iloc[idx]['text'])
            pre.append(transcription[0])

    return pre, ref


def save_predictions(pre, ref):
    idx = 0 
    with open("zeroshot_references.txt", "w") as output:
        # iterate in list and store value in each line
        for row in ref:
            output.write(str(idx)+' '+str(row) + '\n')
            idx += 1

    idx = 0
    #save list as a text file
    with open("zeroshot_predictions.txt", "w") as output:
        # iterate in list and store value in each line
        for row in pre:
            # create a string with the row value
            output.write(str(idx)+' '+str(row) + '\n')
            idx += 1


my_preds, my_refs = generate_predictions(df)
save_predictions(my_preds, my_refs)

import evaluate
wer = evaluate.load('wer')
wer_result = wer.compute(predictions=my_preds, references=my_refs)

print(f'WER: {wer_result}')