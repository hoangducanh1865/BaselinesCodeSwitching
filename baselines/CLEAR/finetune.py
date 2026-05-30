import warnings
warnings.filterwarnings("ignore")
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from datasets import Audio
import torch
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, WhisperForConditionalGeneration, WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor
from utils_prompt import DataCollatorWithPadding
from data.dataloader import PromptWhisperDataset, CodeMixedWhisperDataset_V2
import os
torch.manual_seed(1004)
torch.cuda.manual_seed_all(1004)
import argparse
import evaluate

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser(description='whisper prompt tuning')

    parser.add_argument('--exp-name', type=str, default="test", help="path to save result")
    parser.add_argument('--model', type=str, default="small", help="path to save result")
    parser.add_argument('--batch', type=int, default=16, help="batch size")
    parser.add_argument('--epoch', type=int, default=10, help="batch size")
    parser.add_argument('--lr', type=float, default=1e-4, help="learning rate") #1e-5
    parser.add_argument('--prompt', action='store_true', help="whether to use prompt to decoder")
    parser.add_argument('--dataset', type=str, default="ocw", help="path to save result")
    parser.add_argument('--freeze', action='store_true', help="whether to freeze whisper")

    parser.add_argument('--eval', action='store_true', help="only evaluation")
    parser.add_argument('--model-path', type=str, default="model_path", help="path to save result")  #added by me
    parser.add_argument('--random', action='store_true', help="context perturbation")
    parser.add_argument('--basic', action='store_true', help="collected description")
    args = parser.parse_args()

    # print(args)
    # exit(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cuda")
    
    print("Device:", device)
    args.prompt = True
    
    # prepare feature extractor, tokenizer
    feature_extractor = WhisperFeatureExtractor.from_pretrained(f'openai/whisper-{args.model}')
    tokenizer = WhisperTokenizer.from_pretrained(f'openai/whisper-{args.model}', language='hi', task='transcribe')
    processor = WhisperProcessor.from_pretrained(f'openai/whisper-{args.model}', language='hi', task='transcribe')
    
    data_root = "./data"
    
    if args.dataset == 'earning':
        data_train = PromptWhisperDataset(base_path=os.path.join(data_root,"Earnings_Call/"), phase='train', feature_extractor=feature_extractor, audio_type=".mp3", tokenizer=tokenizer, prompt=args.prompt, random=args.random)
        data_eval = PromptWhisperDataset(base_path=os.path.join(data_root,"Earnings_Call/"), phase='dev', feature_extractor=feature_extractor, audio_type=".mp3", tokenizer=tokenizer, prompt=args.prompt, basic=args.basic)
        data_test = PromptWhisperDataset(base_path=os.path.join(data_root,"Earnings_Call/"), phase='test', feature_extractor=feature_extractor, audio_type=".mp3", tokenizer=tokenizer, prompt=args.prompt, basic=args.basic)

    elif args.dataset == 'ocw':
        data_train = PromptWhisperDataset(base_path=os.path.join(data_root,"ocw/"), phase='train', feature_extractor=feature_extractor, audio_type=".mp3", tokenizer=tokenizer, prompt=args.prompt, basic=args.basic, random=args.random)
        data_eval = PromptWhisperDataset(base_path=os.path.join(data_root,"ocw/"), phase='dev', feature_extractor=feature_extractor, audio_type=".mp3", tokenizer=tokenizer, prompt=args.prompt, basic=args.basic)
        data_test = PromptWhisperDataset(base_path=os.path.join(data_root,"ocw/"), phase='test', feature_extractor=feature_extractor, audio_type=".mp3", tokenizer=tokenizer, prompt=args.prompt, basic=args.basic)
    
    elif args.dataset == 'codemixed':
        data_train = CodeMixedWhisperDataset_V2(csv_path=os.path.join(data_root,"train.csv"), feature_extractor=feature_extractor, tokenizer=tokenizer)
        data_eval = CodeMixedWhisperDataset_V2(csv_path=os.path.join(data_root,"test.csv"), feature_extractor=feature_extractor, tokenizer=tokenizer)
        data_test = CodeMixedWhisperDataset_V2(csv_path=os.path.join(data_root,"test.csv"), feature_extractor=feature_extractor, tokenizer=tokenizer)

    else:
        raise ValueError("Wrong dataset")
    

    metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}



    # load model
    model = WhisperForConditionalGeneration.from_pretrained(f'openai/whisper-{args.model}')
    # Freeze all parameters
    for name, param in model._named_members(lambda module: module._parameters.items()):
        if args.freeze: 
            param.requires_grad = False
        else:
            param.requires_grad = True

    for name, module in model.named_modules():
        if 'decoder' in name:
            for param in module.parameters():
                param.requires_grad = True
    

    model.to(device)
    model.config.forced_decoder_ids = None
    # data collator  
    data_collator = DataCollatorWithPadding(processor=processor, decoder_start_token_id=model.config.decoder_start_token_id)

    root_path = "finetuning_10epoch/"
    os.makedirs(os.path.join(root_path, args.exp_name), exist_ok=True)

    iteration_steps = int(len(data_train) * args.epoch // args.batch)

    eval_step = int((len(data_train) // 2) // args.batch)
    log_step = int((len(data_train) // 50) // args.batch)


    print("Train data len:", len(data_train))
    print("Eval data len:", len(data_test))
    # print("Test data len:", len(data_test))

    print("Max steps:", iteration_steps)
    print("eval step:", eval_step)
    print("log step:", log_step)
    
    
    training_args = Seq2SeqTrainingArguments(
        output_dir = root_path,  # change to a repo name of your choice
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
        learning_rate=args.lr,
        warmup_steps=100,
        max_steps=iteration_steps,
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=8,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=eval_step,
        eval_steps=eval_step,
        logging_steps=log_step,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=data_train,
        eval_dataset=data_test,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    trainer.train()