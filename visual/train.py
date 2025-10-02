import importlib
import os
os.environ["WANDB_DISABLED"] = "true"
from datasets import load_dataset
import torch
from peft import LoraConfig, get_peft_model
from trl import SFTConfig, SFTTrainer
import warnings
warnings.filterwarnings("ignore")
from transformers import AutoProcessor, CLIPVisionModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from vlm_processor import VLMProcessor
import my_vlm
importlib.reload(my_vlm)
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
NUM_STEPS = (283 // BATCH_SIZE) * EPOCHS # this dataset has 283 datapoints
print(f"NUM_STEPS: {NUM_STEPS}")

system_message = """You are a highly advanced Vision Language Model (VLM), specialized in analyzing, describing, and interpreting visual data.
Your task is to process and extract meaningful insights from images, videos, and visual patterns,
leveraging multimodal understanding to provide accurate and contextually relevant information."""

#============== DATA PREPROCESSING ==============#
# for every data point it needs to be in a format the VLM can take in
def format_data(sample):
    '''
    basically turns the datasets from dataset class type to a list type, while mainting the same
    easy access format of the data itself
    '''
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
                                                         split=["train[:1%]", "val[:1%]", "test[:1%]"])

# format the data with the above function
train_dataset = [format_data(sample) for sample in train_dataset]
eval_dataset = [format_data(sample) for sample in eval_dataset]
test_dataset = [format_data(sample) for sample in test_dataset]


#===================== LOAD IN MODEL =====================#
# make apply chat template its own function, necessary for processing the dataset
# load ONLY the tokenizer for the template
tok = AutoTokenizer.from_pretrained(MODEL_ID)

# alias so you can call it like a free function
apply_chat_template = tok.apply_chat_template

# load in my own model
tokenizer  = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
visual_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
LanguageModel=AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
VisionModel = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
processor = VLMProcessor(tokenizer, VisionModel, visual_processor)

# make VLM, config defined by above components
LanguageModel.resize_token_embeddings(len(tokenizer))  # resize to account for new vocab added
my_vlm_config = VLMConfig(VisionModel=VisionModel, LanguageModel=LanguageModel, TextTokenizer=tokenizer) 

# need to pass this tokenizer back in now that it's been modified with new vocab for the processor
vlm = VLM(my_vlm_config, VisionModel=VisionModel, LanguageModel=LanguageModel, TextTokenizer=tokenizer)
model = vlm


#=================== PEFT / LORA ====================#
# the number of params increases, we also freeze parts of the model so only some of the params available should be trainable
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=8,
    bias="none",
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)

peft_model = get_peft_model(model, peft_config)
peft_model.print_trainable_parameters() # After LoRA trainable parameters increases. Since we add adapter.

for param in peft_model.base_model.model.multi_modal_projector.parameters():
    param.requires_grad = True

# After unfreezing projector
for param in peft_model.base_model.model.multi_modal_projector.parameters():
    param.requires_grad = True

# Also unfreeze embedding layer
peft_model.base_model.model.LanguageModel.model.embed_tokens.weight.requires_grad = True

# And lm_head if it exists
if hasattr(peft_model.base_model.model.LanguageModel, 'lm_head'):
    for param in peft_model.base_model.model.LanguageModel.lm_head.parameters():
        param.requires_grad = True

# we set all of these args at the start

#====================== ACTUAL TRAINING =====================#
def collate_fn(examples):
    '''
    will return input ids, attention mask, labels, pixel values
    note that we manually added the labels in this case
    '''
    texts = [apply_chat_template(example, tokenize=False) for example in examples]
    # print(f"texts: {texts}")
    # access the image
    
    image_inputs = [example[1]["content"][0]["image"] for example in examples]
    # print(f"image inputs: {image_inputs}")

    # put our text, image inputs, return tensors and padding since we have batches
    # lists of texts and images
    batch = processor(texts, image_inputs)

    # add labels and put in the pads
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch["labels"] = labels

    return batch


training_args = SFTConfig(
    output_dir="./vlm_training_output", # where checkpoints and models are stored
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_checkpointing=GRADIENT_CHECKPOINTING,
    learning_rate=LEARNING_RATE,
    logging_steps=LOGGING_STEPS,
    eval_steps=EVAL_STEPS,
    eval_strategy=EVAL_STRATEGY,
    save_strategy=SAVE_STRATEGY,
    save_steps=SAVE_STEPS,
    metric_for_best_model=METRIC_FOR_BEST_MODEL,
    load_best_model_at_end=LOAD_BEST_MODEL_AT_END,
    max_grad_norm=MAX_GRAD_NORM,
    warmup_steps=WARMUP_STEPS,
    dataset_kwargs=DATASET_KWARGS,
    max_length=MAX_SEQ_LEN,
    remove_unused_columns = REMOVE_UNUSED_COLUMNS,
    optim=OPTIM,
    bf16=False,  # CPU safety line
    fp16=False,  # Add this too to be safe on CPU
)


trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collate_fn,
    processing_class=processor.tokenizer, # processing just the tokenizer?
)

# note we pass in the regular model, not the peft model as trainer will handle this itself
# evaluation before training is done
print("-"*30)
print("Initial Evaluation")
metric = trainer.evaluate()
print(metric)
print("-"*30)

# evaluation after training
print("Training")
trainer.train()
print("-"*30)

# remember we already set this to save best model, so this will work automatically to do that
trainer.save_model(training_args.output_dir)
