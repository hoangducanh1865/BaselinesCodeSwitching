import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # earlier 0
import torch
import pandas as pd
from transformers import WhisperForConditionalGeneration, WhisperProcessor, GPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer
import torchaudio
from tqdm import tqdm
from transformers import BitsAndBytesConfig
import argparse
import openai
import time
import random


# Set your OpenAI API key

parser = argparse.ArgumentParser(description='Hypotheses Rescoring')
parser.add_argument('--rescorer', type=str, default="gpt2", help="Rescorer model to use (gpt2, llama, deepseek, qwen, mistral)")
parser.add_argument('--input-csv', type=str, default="/home/aseems/Improving-ASR-with-LLM-Description/data/blind_data_mod.csv", help="Path to the input CSV file")
parser.add_argument('--output-csv', type=str, default="./rescored_new_gpt4_beam_5_2_2.csv", help="Path to the output CSV file")
parser.add_argument('--model-path', type=str, default="./results_15epoch/results/test/checkpoint-31312", help="Path to the model checkpoint")
parser.add_argument('--model-type', type=str, default="openai/whisper-small", help="Model type (whisper, gpt2, llama, deepseek, qwen, mistral)")
parser.add_argument('--beam-size', type=int, default=5, help="Beam size for rescoring")
parser.add_argument('--temperature', type=float, default=0.6, help="Temperature for generation")
args = parser.parse_args()


whisper_model = WhisperForConditionalGeneration.from_pretrained(args.model_path).to('cuda')
whisper_processor = WhisperProcessor.from_pretrained(args.model_type)


if args.rescorer == "gpt2":
    rescorer_model = GPT2LMHeadModel.from_pretrained("gpt2").to('cuda')
    rescorer_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
elif args.rescorer == "llama":
    rescorer_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B",).to('cuda')
    rescorer_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
elif args.rescorer == "deepseek":
    rescorer_model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B",).to('cuda')
    rescorer_tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
elif args.rescorer == "qwen":
    rescorer_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-7B",).to('cuda')
    rescorer_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B")
elif args.rescorer == "mistral":
    rescorer_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1",).to('cuda')
    rescorer_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

# old api key = REMOVED_OPENAI_KEY
elif args.rescorer == "gpt3.5":
    openai.api_key = "REMOVED_OPENAI_KEY"



# Load the CSV file with blind data
input_csv_path = args.input_csv
output_csv_path = args.output_csv

# Read the CSV file
df = pd.read_csv(input_csv_path)[2819:]

# Function to generate top-k candidates using Whisper
def generate_candidates(input_path, num_beams=5, num_return_sequences=5, temperature=0.6):
    audio, sr = torchaudio.load(input_path)
    input_ids = whisper_processor(audio[0], sampling_rate=16000, return_tensors='pt').input_features.to('cuda')

    with torch.no_grad():
        candidates = whisper_model.generate(
            input_ids,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            # early_stopping=True,
        )
    decoded_candidates = whisper_processor.batch_decode(candidates, skip_special_tokens=True)
    return decoded_candidates


def rescore_candidates(candidates):
    best_candidate = None
    best_score = -float("inf")
    for candidate in candidates:
        inputs = rescorer_tokenizer(candidate, return_tensors="pt").to('cuda')
        input_ids = inputs.input_ids
        with torch.no_grad():
            outputs = rescorer_model(**inputs)
            logits = outputs.logits

       # Convert logits to log probabilities
        log_probs = torch.log_softmax(logits, dim=-1)

        # Extract log probabilities of the actual token sequence
        token_log_probs = log_probs[0, torch.arange(input_ids.shape[1]), input_ids[0]]

        # Sum over token probabilities
        score = token_log_probs.sum().item()

        if score > best_score:
            best_score = score
            best_candidate = candidate

    return best_candidate

def rescore_candidates_using_openai(candidates):
    best_candidate = None
    best_score = -float("inf")

    for candidate in candidates:
        try:
            response = openai.Completion.create(
                model="gpt-3.5-turbo-instruct",  # logprobs is not available in GPT-4
                prompt=candidate,
                max_tokens=1,  # We don't need completion, just log probs
                logprobs=1,
            )

            # Extract log probabilities for the input tokens
            token_log_probs = response["choices"][0]["logprobs"]["token_logprobs"]
            score = sum(token_log_probs)  # Sum log probabilities for all tokens
            if score > best_score:
                best_score = score
                best_candidate = candidate

        except Exception as e:
            print(f"Error while rescoring: {e}")
            continue

    return best_candidate


def rescore_candidates_using_openai_4(candidates, reference_text):
    best_candidate = None
    best_score = -float("inf")
    api_call_count = 0
    api_call_time = 0
    for candidate in candidates:
        try:
            start_time = time.time()
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an ASR rescoring model. Your task is to rank the given transcriptions based on their accuracy and fluency."},
                    {"role": "user", "content": f"Reference: {reference_text}\nCandidate: {candidate}\nRate the accuracy of this transcription on a scale from -10 (very inaccurate) to 10 (perfectly accurate). Return only the score, no explanations."}
                ]
            )

            end_time = time.time()
            call_duration = end_time - start_time
            api_call_time += call_duration

            # Extract score from GPT-4 response
            score = float(response["choices"][0]["message"]["content"].strip())

            if score > best_score:
                best_score = score
                best_candidate = candidate
            
            api_call_count+=1

            # Apply a sleep timer after every 10 API calls
            if api_call_count % 10 == 0:
                sleep_time = random.uniform(5, 10)  # Random sleep time between 1 to 5 seconds
                print(f"Sleeping for {sleep_time:.2f} seconds to avoid rate limits...")
                time.sleep(sleep_time)

        except Exception as e:
            print(f"Error while rescoring: {e}")
            raise

    return best_candidate, api_call_count, api_call_time


# Process each row in the CSV file
results = []
total_gpt_4_time = 0 # track total time spent on GPT-4-o-mini calls
total_api_calls = 0 # track total number of API calls
try:
    for row in tqdm(df.iterrows(), total=len(df), desc="Processing audio files"):

        audio_path = row[1]["path"]
        input_text = row[1]["text"]

        # Step 1: Generate top-k candidates using Whisper
        candidates = generate_candidates(audio_path, num_beams=args.beam_size, num_return_sequences=args.beam_size, temperature=args.temperature)
        if candidates[0] != input_text:
            # Step 2: Rescore candidates using GPT-2
            if args.rescorer == "gpt3.5":
                best_candidate, calls_made, time_spent = rescore_candidates_using_openai_4(candidates, input_text)
                total_gpt_4_time += time_spent
                total_api_calls += calls_made
            else:
                best_candidate = rescore_candidates(candidates)
        else:
            best_candidate = candidates[0]

        # Save the result
        results.append({
            "ref_text": input_text,
            "best_candidate": best_candidate
        })
        
except Exception as e:
    print(f"An error occurred: {e}. Saving results collected so far.")
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv_path, index=False)
    print(f"Partial results saved to {output_csv_path}.")
    exit(1)

# Save the results to a new CSV file
results_df = pd.DataFrame(results)
results_df.to_csv(output_csv_path, index=False)

print(f"Rescoring complete. Results saved to {output_csv_path}.")
print(f"Total GPT-4o-mini API time: {total_gpt_4_time:.2f} seconds")
print(f"Total GPT-4o-mini API calls: {total_api_calls}")
print(f"Average time per API call: {(total_gpt_4_time/total_api_calls if total_api_calls > 0 else 0):.2f} seconds")