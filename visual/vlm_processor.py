from transformers import AutoProcessor, CLIPVisionModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image


class VLMProcessor:
    '''
    need to get an image and a prompt and convert that into a tensor of 
    <img> tokens + the input ids of the prompt
    final output should be <img> + <bos> + <prompt> + \n
    ALL of this will be in number form so that the output is a tensor
    AND will also include pixel values
    - this means we will have a tokenizer
    - number of image tokens, should be 50 in our case

    - should be handled by the CLIP processor

    all it really needs to handle is the additional image tokens
    NOTE: when this function is returned pixel values will NOT be reshaped yet and input
    ids will just be a tensor. 
    '''

    IMAGE_TOKEN = "<image>"
    BOS_TOKEN = "<bos>"
    PAD_TOKEN = "<pad>"

    def __init__(self, tokenizer, visual_model, visual_processor):
        '''
        sets the tokenizer and processor to be used
        makes a chunk of placeholder image tokens to be added to every prompt
        '''

        # define the length of the image tokens based on the model params
        num_image_tokens = visual_model.vision_model.embeddings.position_embedding.weight.shape[0]
        # add the special token 
        special_tokens_dict = {"additional_special_tokens": [self.IMAGE_TOKEN, self.BOS_TOKEN, self.PAD_TOKEN]}
        tokenizer.add_special_tokens(special_tokens_dict)
        tokenizer.pad_token = self.PAD_TOKEN

        # setup the chunk of tokenized image_tokens to add
        # make num image tokens amount of image tokens
        image_tokens_list = [self.IMAGE_TOKEN] * num_image_tokens
        # convert to a big string
        self.image_tokens_chunk = "".join(image_tokens_list) + self.BOS_TOKEN
        # print(f"image tokens chunk: {self.image_tokens_chunk}")
        # self.image_token_chunk = tokenizer.convert_tokens_to_ids(image_tokens_list) # this function expects a list of tokens

        # define essentials
        self.tokenizer = tokenizer
        self.visual_processor = visual_processor
        # [50, 768] is shape so need to get just the 50

    def __call__(self, texts, images=None, padding=True):
        '''
        call allows this to be used as a function
        takes in the prompt and image and returns the pixel values as
        well as the tokenized version of the image + prompt

        note that we need to be able to take a list of text and a list of images to make batches
        '''
        if images == None:
            full_prompts = [(self.image_tokens_chunk + text + '\n') for text in texts]

            outputs = self.tokenizer(full_prompts, return_tensors="pt", padding=padding)
            return outputs
        
        # get the pixel values in the shape [batch size, num channels, height, width] or width height idk
        pixel_values = self.visual_processor(images=images, return_tensors="pt")
        
        # concatenate the chunk of tokens with the prompt
        full_prompts = [(self.image_tokens_chunk + text + '\n') for text in texts]
        
        # tokenize the whole thing
        outputs = self.tokenizer(full_prompts, return_tensors="pt", padding=padding)
        # add pixel values to the final output
        outputs["pixel_values"] = pixel_values['pixel_values']
        
        return outputs

# testing if collate works
def collate(examples, processor):
    '''
    make sure the processor works in a way that creates batches
    '''
    texts = [example['text'] for example in examples]
    images = [example['image'] for example in examples]
    
    # lists of texts and images
    batch = processor(texts=texts, images=images)

    # add labels and put in the pads
    labels = batch["input_ids"].clone()
    print(f"pad token id: {processor.tokenizer.pad_token_id}")
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch["labels"] = labels

    return batch


def main():
    # test the output of this processor
    text1 = "what do you think that this is a picture of?"
    text2 = "what do you see?"
    image1 = Image.open("./images/laundry.webp")
    image2 = Image.open("./images/dog.webp")

    # make examples
    example1 = {"text": text1, "image": image1}
    example2 = {"text": text2, "image": image2}

    # examples
    examples = [example1, example2] # simulating a batch size of two

    tokenizer  = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    visual_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
    visual_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

    processor = VLMProcessor(tokenizer, visual_model, visual_processor)

    batch = collate(examples=examples, processor=processor)
    print(f"batch: {batch}")

    processed = processor([text1, text2], [image1, image2])
    print(f"processed shape: {processed['pixel_values'].shape}")
    print(f"input id shape: {processed['input_ids'].shape}")

    print(f'pad token id: {tokenizer.pad_token_id}')
    print(f'eos token id: {tokenizer.eos_token_id}')
    print(f'bos token id: {tokenizer.bos_token_id}')

if __name__ == "__main__":
    main()