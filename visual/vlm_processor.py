
class VLMProcessor:
    '''
    need to get an image and a prompt and convert that into a tensor of 
    <img> tokens + the input ids of the prompt
    final output should be <img> + <bos> + <prompt> + \n
    ALL of this will be in number form so that the output is a tensor
    - this means we will have a tokenizer
    - number of image tokens, should be 50 in our case

    will also have to rescale and resize the image into the proper dimensions
    
    '''

    IMAGE_TOKEN = "<image>"



def main():
    # test the output of this processor
    prompt = "this is a test"
    tokenizer  = ""
    image = ""





if __name__ == "__main__":
    main()