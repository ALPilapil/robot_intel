import os
os.environ["WANDB_DISABLED"] = "true"
# disable weights and biases

from datasets import load_dataset
import torch
# using the following modle
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor, BitsAndBytesConfig
# LoRA helps deal with large models, allows us to put LoRA adapaters into model
from peft import LoraConfig, get_peft_model
# get the config and trainer which we need for this
from trl import SFTConfig, SFTTrainer

import warnings
warnings.filterwarnings("ignore")

# my model things
from transformers import AutoProcessor, CLIPVisionModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from vlm_processor import VLMProcessor
from my_vlm import VLM, VLMConfig

#============ HYPER PARAMS ============#
# notice the similarities to the Trainer API
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
EPOCHS = 1
BATCH_SIZE = 1
GRADIENT_CHECKPOINTING = True,  # Tradeoff between memory efficiency and computation time.
USE_REENTRANT = False,
OPTIM = "paged_adamw_32bit"
LEARNING_RATE = 2e-5
LOGGING_STEPS = 50 # how many steps it logs the loss
EVAL_STEPS = 50
SAVE_STEPS = 50
EVAL_STRATEGY = "steps"  # strategies according to steps
SAVE_STRATEGY = "steps"
METRIC_FOR_BEST_MODEL="eval_loss" # save the model with the best loss, not necessarily the last model
LOAD_BEST_MODEL_AT_END=True
MAX_GRAD_NORM = 1
WARMUP_STEPS = 0 # up to us to do some if we want
DATASET_KWARGS={"skip_prepare_dataset": True} # We have to put for VLMs, beacuse we do it ourselves
REMOVE_UNUSED_COLUMNS = False # VLM thing
MAX_SEQ_LEN=128 # for producing text
NUM_STEPS = (50 // BATCH_SIZE) * EPOCHS # this dataset has 283 datapoints
print(f"NUM_STEPS: {NUM_STEPS}")

system_message = """You are a highly advanced Vision Language Model (VLM), specialized in analyzing, describing, and interpreting visual data.
Your task is to process and extract meaningful insights from images, videos, and visual patterns,
leveraging multimodal understanding to provide accurate and contextually relevant information."""

# for every data point it needs to be in a format the VLM can take in
def format_data(sample):
    return [ # list of dicts
        { # first dictionary
            "role": "system",   # the image system prompt to guide the model
            "content": [{"type": "text", "text": system_message}],
        },
        { # second dictionary, user data and content
            "role": "user", # this is the actual input
            "content": [
                {
                    "type": "image",
                    "image": sample["image"],  # the image part of the sample we provide
                },
                {
                    "type": "text",   # the querry provided by the user
                    "text": sample["query"],
                },
            ],
        },
        { # third dictionary
            "role": "assistant",   # the answer from the assistant
            "content": [{"type": "text", "text": sample["label"][0]}], # 0 index of the label, this is just because of how the data is formatted originally
        },
    ]

train_dataset, eval_dataset, test_dataset = load_dataset("HuggingFaceM4/ChartQA",
                                                         split=["train[:50]", "val[:5%]", "test[:5%]"])
print("Before formatting")
print(len(train_dataset))
print("-"*30)
print(train_dataset)
print("-"*30)
print(type(train_dataset))
print("-"*30)
print(train_dataset[0]) # one sample from the dataset
# each on of this gets put into the format function
print("-"*30)

# format the data with the above function
train_dataset = [format_data(sample) for sample in train_dataset]
eval_dataset = [format_data(sample) for sample in eval_dataset]
test_dataset = [format_data(sample) for sample in test_dataset]

print("After formatting:")
print(len(train_dataset))
print("-"*30)
print(type(train_dataset))
print("-"*30)
print(train_dataset[0])
print("-"*30)
print(len(train_dataset[0]))

# basically turns the datasets from dataset class type to a list type, while mainting the same
# easy access format of the data itself

# example of how to access the data, use the function as a "manual" of how to access
# each part of each dataset element
sample_data = test_dataset[0]
sample_question = test_dataset[0][1]['content'][1]['text']
sample_answer = test_dataset[0][2]['content'][0]['text']
sample_image = test_dataset[0][1]['content'][0]['image']

print(sample_question)
print(sample_answer)

# making sure the device is correct
if device == "cuda":
  # quanization here
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        device_map="auto",
        quantization_config=bnb_config,
        use_cache=False
        )

else:
  # no quantization if we're on cpu
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        use_cache=False
        )

# make apply chat template its own function
# load ONLY the tokenizer for the template
tok = AutoTokenizer.from_pretrained(MODEL_ID)

# alias so you can call it like a free function
apply_chat_template = tok.apply_chat_template