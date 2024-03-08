import argparse
import torch 

import numpy as np
import transformers
import wandb

from torch.utils.data import DataLoader
import datasets

from datasets import load_dataset, load_metric, load_dataset_builder, get_dataset_split_names, get_dataset_config_names
from transformers import Trainer, XGLMTokenizer, XGLMTokenizerFast, XGLMForCausalLM, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
# from sklearn.model_selection import train_test_split

MODEL_NAME = "facebook/xglm-564M"
BATCH_SIZE = 2
MAX_LENGTH = 64
EPOCHS = 5

########################################################
# Entry point
########################################################

if __name__ == "__main__":
    # TODO: your code goes here
    # load huggingface quechua dataset

    dataset_qu = load_dataset("wikipedia", language="qu", date="20231020", beam_runner="DirectRunner", split="train[:5000]")
    # dataset_qu = {"train": data["train"][:5000]}
    # dataset_qu = data["train"][:5000]
    
    # print(get_dataset_split_names(dataset_qu, "qu"))
    print(dataset_qu.features)
    print(dataset_qu.column_names)
    print(dataset_qu.shape)
    print(len(dataset_qu))

    # split the dataset into train, and eval
    data = dataset_qu.train_test_split(test_size=0.2, shuffle=True)
    # .train_test_split(dataset_qu["train"][:5000], test_size=0.2, random_state=42)
    print(data["train"].shape)
    print(data["test"].shape)
    

    # load the xglm model
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)


    # finetune the model on quechua dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
    
    train_dataset = data["train"].map(tokenize_function, batched=True)
    eval_dataset = data["test"].map(tokenize_function, batched=True)


    # train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'input_ids'])
    # eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'input_ids'])
    

    # add labels to the dataset from input_ids
    def add_labels(example):
        return {**example, "labels": example["input_ids"]}


    train_dataset = train_dataset.map(add_labels)
    eval_dataset = eval_dataset.map(add_labels)

    print(train_dataset.column_names)


    # define the training arguments
    training_args = transformers.TrainingArguments(
        output_dir="./results",
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100, 
        eval_accumulation_steps=1,
    )

    # define the trainer and train the model on cuda
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=transformers.default_data_collator,
        compute_metrics="accuracy",
    )
    
    # train the model on cuda

    for epoch in range(training_args.num_train_epochs):
        # Train model for one epoch
        trainer.train()

        # Perform memory cleanup
        torch.cuda.empty_cache()

        # Optionally, evaluate or validate the model after each epoch
        metrics = trainer.evaluate()
        print(metrics)

    # Final memory cleanup
    torch.cuda.empty_cache()
    # trainer.train()
    trainer.save_model("./results")



    # log the results to wandb
    wandb.init(project="nnti-project")
    wandb.log({"accuracy": trainer.evaluate()["eval_accuracy"]})
