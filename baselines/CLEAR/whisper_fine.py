import warnings
warnings.filterwarnings("ignore")
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from datasets import Audio
import torch
from transformers_prompt import Seq2SeqTrainingArguments, Seq2SeqTrainer, WhisperPromptForConditionalGeneration, GenerationConfig, WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor
from utils_prompt import compute_wer, DataCollatorSpeechS2SWhitPadding
from data.dataloader import PromptWhisperDataset, CodeMixedWhisperDataset
import os
torch.manual_seed(1004)
torch.cuda.manual_seed_all(1004)
import argparse

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    parser = argparse.ArgumentParser(description='whisper prompt tuning')

    parser.add_argument('--exp-name', type=str, default="test", help="path to save result")
    parser.add_argument('--model', type=str, default="small", help="path to save result")
    parser.add_argument('--batch', type=int, default=16, help="batch size")
    parser.add_argument('--epoch', type=int, default=15, help="batch size")
    parser.add_argument('--lr', type=float, default=1e-4, help="learning rate") #1e-5
    parser.add_argument('--prompt', action='store_true', help="whether to use prompt to decoder")
    parser.add_argument('--dataset', type=str, default="ocw", help="path to save result")
    parser.add_argument('--freeze', action='store_true', help="whether to freeze whisper")

    parser.add_argument('--eval', action='store_true', help="only evaluation")
    parser.add_argument('--model-path', type=str, default="model_path", help="path to save result")  #added by me
    parser.add_argument('--random', action='store_true', help="context perturbation")
    parser.add_argument('--basic', action='store_true', help="collected description")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Device:", device)
    args.prompt = True
    print(args)
    # exit(0)
    # prepare feature extractor, tokenizer
    feature_extractor = WhisperFeatureExtractor.from_pretrained(f'openai/whisper-{args.model}')
    tokenizer = WhisperTokenizer.from_pretrained(f'openai/whisper-{args.model}', language='hi', task='transcribe')
    processor = WhisperProcessor.from_pretrained(f'openai/whisper-{args.model}', language='hi', task='transcribe')

    # data collator  
    data_collator = DataCollatorSpeechS2SWhitPadding(processor=processor)
    
    data_root = "./data" #"/home/aseems/Improving-ASR-with-LLM-Description/data"

    # data_root = "/home/aseems/"
    
    if args.dataset == 'earning':
        data_train = PromptWhisperDataset(base_path=os.path.join(data_root,"Earnings_Call/"), phase='train', feature_extractor=feature_extractor, audio_type=".mp3", tokenizer=tokenizer, prompt=args.prompt, random=args.random)
        data_eval = PromptWhisperDataset(base_path=os.path.join(data_root,"Earnings_Call/"), phase='dev', feature_extractor=feature_extractor, audio_type=".mp3", tokenizer=tokenizer, prompt=args.prompt, basic=args.basic)
        data_test = PromptWhisperDataset(base_path=os.path.join(data_root,"Earnings_Call/"), phase='test', feature_extractor=feature_extractor, audio_type=".mp3", tokenizer=tokenizer, prompt=args.prompt, basic=args.basic)

    elif args.dataset == 'ocw':
        data_train = PromptWhisperDataset(base_path=os.path.join(data_root,"ocw/"), phase='train', feature_extractor=feature_extractor, audio_type=".mp3", tokenizer=tokenizer, prompt=args.prompt, basic=args.basic, random=args.random)
        data_eval = PromptWhisperDataset(base_path=os.path.join(data_root,"ocw/"), phase='dev', feature_extractor=feature_extractor, audio_type=".mp3", tokenizer=tokenizer, prompt=args.prompt, basic=args.basic)
        data_test = PromptWhisperDataset(base_path=os.path.join(data_root,"ocw/"), phase='test', feature_extractor=feature_extractor, audio_type=".mp3", tokenizer=tokenizer, prompt=args.prompt, basic=args.basic)
    
    elif args.dataset == 'mucs':
        data_train = CodeMixedWhisperDataset(csv_path=os.path.join(data_root,"train.csv"), dataset_name='mucs', feature_extractor=feature_extractor, tokenizer=tokenizer, prompt=args.prompt, train=True)
        data_eval = CodeMixedWhisperDataset(csv_path=os.path.join(data_root,"test.csv"), dataset_name='mucs', feature_extractor=feature_extractor, tokenizer=tokenizer, prompt=args.prompt, train=False)
        data_test = CodeMixedWhisperDataset(csv_path=os.path.join(data_root,"test.csv"), dataset_name='mucs', feature_extractor=feature_extractor, tokenizer=tokenizer, prompt=args.prompt, train=False, test=True)

    elif args.dataset == 'sarabhai':
        # sarabhai dataset
        data_train = CodeMixedWhisperDataset(csv_path=os.path.join(data_root,"train.csv"), dataset_name='sarabhai', feature_extractor=feature_extractor, tokenizer=tokenizer, prompt=args.prompt, train=True)
        data_eval = CodeMixedWhisperDataset(csv_path=os.path.join(data_root,"val.csv"), dataset_name='sarabhai', feature_extractor=feature_extractor, tokenizer=tokenizer, prompt=args.prompt, train=False)
        data_test = CodeMixedWhisperDataset(csv_path=os.path.join(data_root,"test.csv"), dataset_name='sarabhai', feature_extractor=feature_extractor, tokenizer=tokenizer, prompt=args.prompt, train=False, test=True)
    
    elif args.dataset == 'gramvani':
        data_train = CodeMixedWhisperDataset(csv_path=os.path.join(data_root,"GV_Train_100h/train.csv"), dataset_name='gramvani', feature_extractor=feature_extractor, tokenizer=tokenizer, prompt=args.prompt, train=True)
        data_eval = CodeMixedWhisperDataset(csv_path=os.path.join(data_root,"GV_Dev_5h/dev.csv"), dataset_name='gramvani', feature_extractor=feature_extractor, tokenizer=tokenizer, prompt=args.prompt, train=False)
        data_test = CodeMixedWhisperDataset(csv_path=os.path.join(data_root,"GV_Eval_3h/test.csv"), dataset_name='gramvani', feature_extractor=feature_extractor, tokenizer=tokenizer, prompt=args.prompt, train=False, test=True)


    else:
        raise ValueError("Wrong dataset")


    # load model
    if args.prompt:
        model = WhisperPromptForConditionalGeneration.from_pretrained(f'openai/whisper-{args.model}')
        # Freeze all parameters
        # Note - I will uncomment this part after the rebuttal period.
        for name, param in model._named_members(lambda module: module._parameters.items()):
            if args.freeze: 
                param.requires_grad = False
            else:
                param.requires_grad = True

        for name, module in model.named_modules():
            if 'decoder' in name:
                for param in module.parameters():
                    param.requires_grad = True
    else:
        print("Prompt must be used.")
        raise(ValueError)
    
    if args.eval:
        model = model.from_pretrained(args.model_path)
        print("model loaded!!")
    model.to(device)

    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    # root_path = "prompt_15epoch/"
    # root_path = "/home/aseems/Improving-ASR-with-LLM-Description/whisper_medium_results_2"  #-> running at tmux 5
    # root_path = "/home/aseems/Improving-ASR-with-LLM-Description/whisper_large_results"  # -> running at tmux 4
    # root_path = "/home/aseems/Improving-ASR-with-LLM-Description/few_shot_prompt"  # -> running at tmux 1
    # root_path = "/home/aseems/Improving-ASR-with-LLM-Description/Gramvani_files/results/"
    # root_path = "/home/aseems/Improving-ASR-with-LLM-Description/rebuttal_whisper_unfreezed"  # -> running at tmux 3
    root_path = "./whisper_prompt_2_results"

    os.makedirs(os.path.join(root_path, args.exp_name), exist_ok=True)

    iteration_steps = int(len(data_train) * args.epoch // args.batch)

    eval_step = int((len(data_train) // 2) // args.batch)
    log_step = int((len(data_train) // 50) // args.batch)

    # eval_step = len(data_train)
    # log_step = 1000
    print("Train data len:", len(data_train))
    print("Eval data len:", len(data_eval))
    # print("Test data len:", len(data_test))

    print("Max steps:", iteration_steps)
    print("eval step:", eval_step)
    print("log step:", log_step)
    
    generation_config = GenerationConfig(
        pos_token_id=50360
    )
    
    training_args = Seq2SeqTrainingArguments(
        weight_decay=0.01,
        output_dir= os.path.join(root_path, "results", args.exp_name),  # change to a repo name of your choice
        dataloader_num_workers=1,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
        learning_rate=args.lr,
        warmup_steps=100,
        max_steps=iteration_steps,
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=1,
        predict_with_generate=True,
        generation_max_length=225,
        save_steps=eval_step,
        eval_steps=eval_step,
        logging_steps=log_step,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        dataloader_pin_memory=False,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        model=args.model,
        remove_unused_columns=False,
        pos_token_id=tokenizer.convert_tokens_to_ids("<|startofprev|>")
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=data_train,
        eval_dataset=data_eval,
        data_collator=data_collator,
        compute_metrics=compute_wer,
        tokenizer=processor.feature_extractor,
    )

    if not args.eval:
        print("Start Training!")
        # trainer.train(resume_from_checkpoint = True) # if needed
        trainer.train()

    print("Start Evaluation!!")
    if args.prompt:
        print("Using prompt")
    result = trainer.evaluate(data_test)  # previously it was data_test
    print(result)
    
    # print results
    with open(os.path.join(root_path, "results", args.exp_name, 'result_blind.txt'), 'w') as t:
        t.write(str(result))
