# Databricks notebook source
import os
import time
import datetime

import pandas as pd
import numpy as np
import random

import torch
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
# torch.manual_seed(42)

from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config, GPT2LMHeadModel, GPTNeoConfig, GPTNeoForCausalLM
from transformers import AdamW, get_linear_schedule_with_warmup

import nltk
nltk.download('punkt')
#!nvidia-smi Type in Console

# Load dataset
data = []
with open("/dbfs/FileStore/shared_uploads/ryanp/FRSDictionary_Cleaned.txt", 'r', encoding="utf8") as f1:
    for src in f1:
        data.append(src.strip())
        
data_formatted = []    
temp_list = []
for ele in data:
    if len(ele) > 0:
        temp_list.append(ele)
    else:
        joined_strings = ' '.join(temp_list)
        data_formatted.append(joined_strings)
        temp_list = []
        
# Remove all empty rows of text
data_filtered = list(filter(None, data_formatted))
df_data = pd.DataFrame(data_filtered)

# Load the GPT tokenizer.

# Load a trained model and vocabulary that you have fine-tuned
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
output_dir = '/dbfs/FileStore/shared_uploads/ryanp/FRSCleanedTrainedModel_280Epochs_23_11_2022_2_17'
tokenizer = GPT2Tokenizer.from_pretrained(output_dir)

batch_size = 2

class GPT2Dataset(Dataset):

  def __init__(self, txt_list, tokenizer, gpt2_type="gpt2", max_length=591):

    self.tokenizer = tokenizer
    self.input_ids = []
    self.attn_masks = []

    for txt in txt_list:

      encodings_dict = tokenizer('<|startoftext|>'+ txt + '<|endoftext|>', truncation=True, max_length=max_length, padding="max_length")

      self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
      self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
    
  def __len__(self):
    return len(self.input_ids)

  def __getitem__(self, idx):
    return self.input_ids[idx], self.attn_masks[idx] 


#dataset = GPT2Dataset(df_data.iloc[0:3000,0], tokenizer, max_length=768)
dataset = GPT2Dataset(df_data.iloc[:,0], tokenizer, max_length=591)

# Split into training and validation sets
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

print('{:>5,} training samples'.format(train_size))
print('{:>5,} validation samples'.format(val_size))

# Create the DataLoaders for our training and validation datasets.
# Take training samples in random order. 
train_dataloader = DataLoader(
            train_dataset,  # The training samples.
            sampler = RandomSampler(train_dataset), # Select batches randomly
            batch_size = batch_size # Trains with this batch size.
        )

# For validation the order is not important, so we simply read them sequentially.
validation_dataloader = DataLoader(
            val_dataset, # The validation samples.
            sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
            batch_size = batch_size # Evaluate with this batch size.
        )

# Instantiate the model

# model = GPTNeoForCausalLM.from_pretrained(output_dir)
model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M", config=configuration)

# Randomise weights if desired
# model.init_weights()

# This step is necessary because I've added some tokens (bos_token, etc) to the embeddings
# otherwise the tokenizer and model tensors won't match up
model.resize_token_embeddings(len(tokenizer))

# Can provide an input prompt to the pre-trained model here and obtain an output if desired

# prompt = (
#     "In a shocking finding, scientists discovered a herd of unicorns living in a remote, "
#     "previously unexplored valley, in the Andes Mountains. Even more surprising to the "
#     "researchers was the fact that the unicorns spoke perfect English."
# )

# input_ids = tokenizer(prompt, return_tensors="pt").input_ids

# gen_tokens = model.generate(
#     input_ids,
#     do_sample=True,
#     temperature=0.9,
#     max_length=200,
# )

# gen_text = tokenizer.batch_decode(gen_tokens)[0]
# print(gen_text)


# Tell pytorch to run this model on the GPU.
device = torch.device("cuda")
model.cuda()

# Define a set of initial training parameters

epochs = 20
learning_rate = 5e-4
warmup_steps = 1e2
epsilon = 1e-8

# Produce a sample output every 1000 steps
sample_every = 1000

# Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
optimizer = AdamW(model.parameters(),
                  lr = learning_rate,
                  eps = epsilon
                )

# Total number of training steps is [number of batches] x [number of epochs]. 
# (Note that this is not the same as the number of training samples).
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
# This changes the learning rate as the training loop progresses
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = warmup_steps, 
                                            num_training_steps = total_steps)

def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))

total_t0 = time.time()

training_stats = []

model = model.to(device)

for epoch_i in range(0, epochs):

    # ========================================
    #               Training
    # ========================================

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    t0 = time.time()

    total_train_loss = 0

    model.train()
    
    for step, batch in enumerate(train_dataloader):

        b_input_ids = batch[0].to(device)
        b_labels = batch[0].to(device)
        b_masks = batch[1].to(device)

        model.zero_grad()        

        outputs = model(  b_input_ids,
                          labels=b_labels, 
                          attention_mask = b_masks,
                          token_type_ids=None
                        )

        loss = outputs[0]  

        batch_loss = loss.item()
        total_train_loss += batch_loss

        loss.backward()

        optimizer.step()

        scheduler.step()

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)       
    
    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epoch took: {:}".format(training_time))
        
    # ========================================
    #               Validation
    # ========================================

    print("")
    print("Running Validation...")

    t0 = time.time()

    model.eval()

    total_eval_loss = 0
    nb_eval_steps = 0

    # Evaluate data for one epoch
    for batch in validation_dataloader:
        
        b_input_ids = batch[0].to(device)
        b_labels = batch[0].to(device)
        b_masks = batch[1].to(device)
        
        with torch.no_grad():        

            outputs  = model(b_input_ids, 
#                            token_type_ids=None, 
                             attention_mask = b_masks,
                            labels=b_labels)
          
            loss = outputs[0]  
            
        batch_loss = loss.item()
        total_eval_loss += batch_loss        

    avg_val_loss = total_eval_loss / len(validation_dataloader)
    
    validation_time = format_time(time.time() - t0)    

    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    # Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )

print("")
print("Training complete!")
print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
  
# Save Model in dbfs 
now = datetime.datetime.now()
current_date = str(now.day) + "_" + str(now.month) + "_" + str(now.year) + "_" + str(now.hour) + \
               "_" + str(now.minute)

# If you use default names for the model, you can reload it using from_pretrained()
output_dir = '/dbfs/FileStore/shared_uploads/ryanp\FRSCleanedTrainedModel_300Epochs_' + current_date

# Create output directory if needed
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Saving model to %s" % output_dir)

# Save a trained model, configuration and tokenizer using `save_pretrained()`.
# They can then be reloaded using `from_pretrained()`
model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
model_to_save.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
