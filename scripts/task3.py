import argparse
import os
import torch 

import numpy as np
import transformers
import wandb

from torch.utils.data import DataLoader
import datasets

from datasets import load_dataset, load_metric, load_dataset_builder, get_dataset_split_names, get_dataset_config_names
from transformers import Trainer, XGLMTokenizer, XGLMTokenizerFast, XGLMForCausalLM, AutoModelForCausalLM, AutoTokenizer, GenerationConfig, \
AdamW, get_linear_schedule_with_warmup

from lora import XGLMwithLoRA

from tqdm import tqdm
from datetime import datetime
from functools import partial

from iA3 import modify_model_for_iA3



# from sklearn.model_selection import train_test_split

MODEL_NAME = "facebook/xglm-564M"
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16
MAX_LENGTH = 32
EPOCHS = 2
LR=5e-5
WEIGHT_DECAY=0.01
STRATEGY = "lora" # full, bitfit, lora, iA3
DATASET_SIZE="500" # full or a number
save_dir = "trained_models"

########################################################
# Entry point
########################################################


# preprocess the logits for metrics
def preprocess_logits_for_metrics(logits):
    return torch.argmax(logits, dim=-1)

# add labels to the dataset from input_ids
def add_labels(example):
    return {**example, "labels": example["input_ids"]}


# define the tokenizer function
def tokenize_function(examples):
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    if 'text' in examples.keys():
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    else:
        return tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=MAX_LENGTH, return_tensors="pt")


def test(model, dataloaders, device='cuda'):

    losses = {lang: [] for lang in LANGUAGES}
    with torch.no_grad():
        for lang in LANGUAGES:
            print(lang)
            for batch in tqdm(dataloaders[lang]):
                for key in batch:
                    batch[key] = batch[key].to(device)
                outputs = model(**batch)
                loss = outputs.loss
                losses[lang].append(loss.item())

    return losses


def define_optimizer(model, learning_rate=5e-5, weight_decay=0.01, strategy="full"):
    
    if strategy == "bitfit":
        for name, param in model.named_parameters():
            if "bias" not in name:
                param.requires_grad = False
        model.lm_head.weight.requires_grad = True
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)
        return optimizer
    
    else:
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)
        return optimizer
    

if __name__ == "__main__":
    # TODO: your code goes here
    
    if DATASET_SIZE == "full":
        load_split_size=""
    # elif DATASET_SIZE is a number:
    elif DATASET_SIZE.isdigit():
        load_split_size = f"[:{DATASET_SIZE}]"
    else:
        raise ValueError("DATASET_SIZE should be either 'full' or a number")
    
    # load huggingface quechua dataset (first 5000 examples)
    dataset = load_dataset("wikipedia", language="qu", date="20231020", split=f"train{load_split_size}")
    train_cols = list(dataset.features.keys())

    # tokenize the dataset
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=train_cols)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'], format_kwargs={'dtype': torch.long})
    # add labels to the dataset
    tokenized_dataset = tokenized_dataset.map(add_labels)

    # split the dataset into train, and eval
    train_size = int(0.8 * len(tokenized_dataset))
    eval_size = len(tokenized_dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(tokenized_dataset, [train_size, eval_size])

    # train dataloader and eval dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=EVAL_BATCH_SIZE, shuffle=True, drop_last=True)


    # load the flores dataset
    ds_builder = load_dataset_builder("facebook/flores", "deu_Latn")

    # define evaluation languages from flores
    LANGUAGES = [
    "eng_Latn",
    "spa_Latn",
    "ita_Latn",
    "deu_Latn",
    "arb_Arab",
    "tel_Telu",
    "tam_Taml",
    "quy_Latn",
]

    # load the evaluation dataset
    test_datasets = {}
    tokenized_test_datasets = {}
    test_dataloaders = {}

    for lang in LANGUAGES:
        test_datasets[lang] = load_dataset("facebook/flores", lang)
        tokenized_test_datasets[lang] = test_datasets[lang].map(tokenize_function, batched=True, remove_columns=list(ds_builder.info.features.keys()))
    
        tokenized_test_datasets[lang].set_format(type='torch', columns=['input_ids', 'attention_mask'], format_kwargs={'dtype': torch.long})
        tokenized_test_datasets[lang] = tokenized_test_datasets[lang].map(add_labels)
        test_dataloaders[lang] = DataLoader(tokenized_test_datasets[lang]["devtest"], batch_size=EVAL_BATCH_SIZE, shuffle=True, drop_last=True)

    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # define the model
    if STRATEGY == "lora":
        model = XGLMwithLoRA(MODEL_NAME, rank=16, alpha=16)
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    # if STRATEGY == "lora":
    #     model = modify_model_for_lora(model, rank=16, alpha=16)
    if STRATEGY == "iA3":
        model = modify_model_for_iA3(model)
    
    optimizer = define_optimizer(model, learning_rate=LR, weight_decay=WEIGHT_DECAY, strategy=STRATEGY)
    num_training_steps = EPOCHS*len(train_dataloader)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    model.to(device)
    # log the results to wandb and set name of the run as current timestamp in UTC
    wandb.init(project="nnti-project", name=STRATEGY+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    wandb.config = {
    "learning_rate": LR,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "strategy": STRATEGY,
    }

    # define the training loop
    best_eval_loss = torch.inf
    
    for epoch in range(EPOCHS):
        model.train()
        for iter, batch in tqdm(enumerate(train_dataloader)):
            # labels = batch["input_ids"].to(device)
            for key in batch:
                batch[key] = batch[key].to(device)
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
            if iter % 50 == 0:
                print(f"Epoch: {epoch}, Iteration: {iter}, Training Loss: {loss.item()}")

                # log the results to wandb
                wandb.log({"train_loss": loss.item(), "epoch" : epoch, "step":iter})
                

        # evaluate the model
        eval_losses = []
        model.eval()
        with torch.no_grad():
            for batch in eval_dataloader:
                for key in batch:
                    batch[key] = batch[key].to(device)
                outputs = model(**batch)
                loss = outputs.loss
                eval_losses.append(loss.item())
        
        mean_eval_loss = torch.mean(torch.tensor(eval_losses))
        print(f"Epoch: {epoch}, Evaluation Loss: {mean_eval_loss}")
        wandb.log({"eval_loss": mean_eval_loss, "epoch" : epoch})

        
        
        # test the model on flores (test) dataset after each epoch
        test_losses = test(model, test_dataloaders, device)
        mean_test_losses = {lang: torch.mean(torch.tensor(test_losses[lang])) for lang in LANGUAGES}

        for lang in LANGUAGES:
            print(f"Epoch: {epoch}, Language: {lang}, Test Loss: {mean_test_losses[lang]}")
            wandb.log({f"test_loss_{lang}": mean_test_losses[lang], "epoch" : epoch})

        test_loss = torch.mean(torch.tensor([mean_test_losses[lang] for lang in LANGUAGES]))

        print(f"Epoch: {epoch}, Test Loss: {test_loss}")
        wandb.log({"eval_loss": test_loss, "epoch" : epoch})

        # make a directory to save the model once the first epoch is finished
        if epoch == 0:
            save_dir = f"trained_models/xglm_{STRATEGY}_{load_split_size}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
            os.makedirs(save_dir, exist_ok=True)
    
        model.save_pretrained(f"{save_dir}/epoch_{epoch}_model_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
        
        
        # save the model if the evaluation loss is the best
        if mean_eval_loss < best_eval_loss:
            best_eval_loss = mean_eval_loss
            model.save_pretrained(f"{save_dir}/best_model")
            print("best model saved")
        
    wandb.finish()
    print("Training finished")
