# robot_intel
A general purpose robot utilizing VLMs.

# architecutre
Very similar to the PaliGemma Model. We use a vision encoding model provided by CLIP and a causal language model provided by LLaMA to generate textual output. We train this model on image goal pairs with the output being a set of instructions to carry out this goal. 

POTENTIAL: We also use object detection with OwlViT to see if certain objects are available in the scence that may help the model carry out its goal. Since OwlViT uses an open vocabulary we use the central goal object of the user prompt to then generate a list of highly relevant objects to search for. 