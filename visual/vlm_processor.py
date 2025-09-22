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

    def __init__(self, tokenizer, visual_model, visual_processor):
        '''
        sets the tokenizer and processor to be used
        makes a chunk of placeholder image tokens to be added to every prompt
        '''

        # define the length of the image tokens based on the model params
        num_image_tokens = visual_model.vision_model.embeddings.position_embedding.weight.shape[0]
        # add the special token 
        special_tokens_dict = {"additional_special_tokens": [self.IMAGE_TOKEN, self.BOS_TOKEN]}
        tokenizer.add_special_tokens(special_tokens_dict)

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

    def __call__(self, prompt, image):
        '''
        call allows this to be used as a function
        takes in the prompt and image and returns the pixel values as
        well as the tokenized version of the image + prompt
        '''
        # get the pixel values in the shape [batch size, num channels, height, width] or width height idk
        pixel_values = self.visual_processor(images=image, return_tensors="pt")
        
        # concatenate the chunk of tokens with the prompt
        full_prompt = self.image_tokens_chunk + prompt + '\n'
        
        # tokenize the whole thing
        outputs = self.tokenizer(full_prompt, return_tensors="pt")
        outputs["pixel_values"] = pixel_values['pixel_values']
        
        return outputs


def main():
    # test the output of this processor
    prompt = "this is a test"
    tokenizer  = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    image = Image.open("./images/laundry.webp")
    visual_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
    visual_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

    processor = VLMProcessor(tokenizer, visual_model, visual_processor)

    processed = processor(prompt, image)
    print(f"processed shape: {processed['pixel_values'].shape}\nprocessed: {processed}")

if __name__ == "__main__":
    main()