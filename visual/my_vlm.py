from PIL import Image
import requests
import torch
from torch import nn
from transformers import AutoProcessor, CLIPVisionModel
from transformers import AutoTokenizer, AutoModelForCausalLM


def vision_encoder():
    '''    
    '''
    model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

    print(f"Model config: {model.config}")
    print(f"Hidden size: {model.config.hidden_size}")
    print(f"model: {type(model)}")

    # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    # image = Image.open(requests.get(url, stream=True).raw)

    # # processor takes an image and returns the pixel values in [batch_size, channels, height, width]
    # inputs = processor(images=image, return_tensors="pt") 
    # # image can be a list or just one, batch size will match this
    # print(f"inputs type: {type(inputs['pixel_values'])} \n inputs: {inputs['pixel_values'].shape}")

    # outputs = model(**inputs) # of type transformer class, ** unpacks it, essentially doing what's done above in the with the dict
    # last_hidden_state = outputs.last_hidden_state # extracts the actual tensor from the above variable
    # pooled_output = outputs.pooler_output  # pooled CLS states, a single vector representation that summarizes entire input, this is what we want to use

    # print(f"last_hidden_state shape: {last_hidden_state.shape}") # [batch size, seq length, embed dim], batch size is how many images are passed through, seq length = how many chunks image is divided into 
    # print(f"pooled_output shape: {pooled_output.shape}") # [batch size, embed dim]

# TEXT MODEL
def llamma():
    '''
    embed dim of 2024
    '''
    # Load model and tokenizer
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    # print(f"model config: {model.config}")
    # print(f"model: {type(model)}")
    # print(f"hidden size: {model.config.hidden_size}")

    # Add padding token
    tokenizer.pad_token = tokenizer.eos_token

    # Test generation
    prompt = "please Explain what a cat is:"
    encoded_input = tokenizer(prompt, return_tensors="pt")
    print(f"inputs: {encoded_input}")

    # Extract embeddings manually
    input_ids = encoded_input['input_ids']
    inputs_embeds = model.model.embed_tokens(input_ids)  # Access embedding layer
    print(f"Embeddings shape: {inputs_embeds.shape}")
    print(f"Embedding dimension: {inputs_embeds.shape[-1]}")
    
    # You can now pass either input_ids OR inputs_embeds to the model
    # Option 1: Using input_ids (what you're currently doing)
    output1 = model(**encoded_input)
    
    # Option 2: Using inputs_embeds directly
    output2 = model(inputs_embeds=inputs_embeds, attention_mask=encoded_input['attention_mask'])
    
    print(f"Outputs are equivalent: {torch.allclose(output1.logits, output2.logits)}")

    output = model(**encoded_input)

    # hidden states are here:
    print(f"output type: {type(output)}")
    print(f"available attributes: {dir(output)}")
    
    # Get the actual tensors:
    print(f"logits shape: {output.logits.shape}")           # [batch, seq_len, vocab_size], 
                                                            # batch = amount of items in prompt, seq len = tokens in each one, vocab size is vocab size
    print(f"hidden states: {output.hidden_states}")         # Will be None by default
    print(f"attentions: {output.attentions}")
    
    # To get the actual hidden states, you need to request them:
    output_with_hidden = model(**encoded_input, output_hidden_states=True)
    print(f"hidden states shape: {output_with_hidden.hidden_states[-1].shape}")  # Last layer denoted as -1

    # generating text
    with torch.no_grad():
        outputs = model.generate(
            **encoded_input,
            max_length=100,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        print(f"new outputs: {outputs}")

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"gnerated text: {generated_text}")
    print(f"outputs: {outputs}")
        
        
#----------------- SUB CLASSES -----------------#
class VLMConfig():
    def __init__(self, VisionModel=CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32"),
                 LanguageModel=AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
                 text_tokenizer=AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
                 pad_token_id=None):
        self.VisionModel = VisionModel
        self.LanguageModel = LanguageModel
        self.TextTokenizer = text_tokenizer
        self.vision_config = VisionModel.config
        self.text_config = LanguageModel.config
        self.pad_token_id = pad_token_id

class MultiModalProjector(nn.Module):
    def __init__(self, config: VLMConfig):
        super().__init__()
        self.linear = nn.Linear(config.vision_config.hidden_size, config.text_config.hidden_size)
        # linear layer to convert input from vision hidden size to text hidden size
    
    def forward(self, image_features):
        # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Projection_Dim]
        hidden_states = self.linear(image_features)
        return hidden_states

#----------------- MAIN CLASS -----------------#
class VLM(nn.Module):
    def __init__(self, config: VLMConfig):
        super().__init__()
        self.config = config
        # create an instance of the vision encoder
        self.VisionModel = config.VisionModel
        # creat an instance of the language model and tokenizer
        self.LanguageModel = config.LanguageModel
        self.TextTokenizer = config.TextTokenizer
        # multi modal projector 
        self.multi_modal_projector = MultiModalProjector(config)
        # set the padding token
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

    # def _merge_input_ids_with_image_features(
    #         self, image_features, 
    # ):
    #     '''
    #     returns
    #     '''

    def forward(
            self,
            input_ids, # extracted from the <image> + <bos> + prompt + \n
            pixel_values, # we will get this from the processor
    ):
        # we already have models that return the vectors we need, all we have to do is merge them
        # with a multi modal projector and a merge input ids with image features

        # get image features from the pixel values which we will get from the processor
        # [Batch_Size, Channels, Height, Width] -> [Batch_Size, Num_Patches, Embed_Dim]
        selected_image_feature = self.VisionModel(pixel_values)
        # [batch_size, patch_len, embed_dim]
        last_hidden_state = selected_image_feature.last_hidden_state # extracts the actual tensor from the above variable
        # [batch_size, embed_dim]
        pooled_output = selected_image_feature.pooler_output  # pooled CLS states, a single vector representation that summarizes entire input, this is what we want to use
        # resize image ebeddings into language model embeddings dimensions
        # make their embed dimensions match basically, but this is still the image vectors
        image_features = self.multi_modal_projector(selected_image_feature)
        # merge the embeddings of the text tokens and image tokens, need to figure out what inputs to give it
        

        return 



def main():
    # SO FAR:
    # a text decoder and visual encoder that are of different dimensions
    # vision: 768    text: 2048
    # note that both are [batch, seq_len, embed_dim]

    # TODO:
    # MultiModalProjector
    # tie weights
    # merge input ids with image features
    # - returns inputs_embeds, attention_mask, position_ids
    # language model needs to be constructed in a way to take these inputs

    # vision_encoder()
    llamma()
    # my_vlm_config = VLMConfig()
    # vlm = VLM(my_vlm_config)
    # print(f"vlm vision model config: {vlm.VisionModel.config}")
    # print(f"vlm text model config: {vlm.LanguageModel.config}")





if __name__ == "__main__":
    main()