# this is a vision encoder

from typing import Optional, Tuple
import torch
import torch.nn as nn

class SiglipVisionConfig:

    def __init__(
        self,
        hidden_size=768, # size of the embedding vector 
        intermediate_size=3072,  # size of linear layer in feed forward
        num_hidden_layers=12, # num of layers of visual transformers
        num_attention_heads=12,  # duh
        num_channels=3, # R G B = 3 channels
        image_size=224, # 3 sizes available: 224, 448, 896. img dimension basically
        patch_size=16, # how many patches each image will be divded into
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        num_image_tokens: int = None, # how many output embeddings this transformer will output for each image
        **kwargs
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.num_image_tokens = num_image_tokens


class SiglipVisionEmbeddings(nn.Module):
    # basically identical to config because it needs all those params  
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels, # R G B
            out_channels=self.embed_dim, # hidden size 
            kernel_size=self.patch_size, # the grouping of pixels basically
            stride=self.patch_size, # how to slide kernel from one group to next, will cause no overlap of groupings
            paddings = "valid", # no paddings added
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2 
        # imagine the 2d example, width x heigh in a square
        self.num_positions = self.num_patches
        # each kernel position, since no overlap, will be used for pos encoding
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions).expand((1, -1)),
            persistent=False,
        )

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        _, _, height, width = pixel_values.shape # [Batch_Size, Channels, Height, Width]
        # Convolve the `patch_size` kernel over the image, with no overlapping patches since the stride is equal to the kernel size
        # The output of the convolution will have shape [Batch_Size, Embed_Dim, Num_Patches_H, Num_Patches_W]
        # where Num_Patches_H = height // patch_size and Num_Patches_W = width // patch_size
        patch_embeds = self.patch_embedding(pixel_values)  # defined above as 2d conv
        # [Batch_Size, Embed_Dim, Num_Patches_H, Num_Patches_W] -> [Batch_Size, Embed_Dim, Num_Patches]
        # where Num_Patches = Num_Patches_H * Num_Patches_W
        embeddings = patch_embeds.flatten(2)  # output of 2dconv is a 2d grid but we need it as a flat tensor
        # [Batch_Size, Embed_Dim, Num_Patches] -> [Batch_Size, Num_Patches, Embed_Dim]
        embeddings = embeddings.transpose(1, 2) # num patches must come before so that this becomes a batch of sequence of embeddings
        # Add position embeddings to each patch. Each positional encoding is a vector of size [Embed_Dim]
        embeddings = embeddings + self.position_embedding(self.position_ids)
        # [Batch_Size, Num_Patches, Embed_Dim]
        return embeddings

class SiglipAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5 # Equivalent to 1 / sqrt(self.head_dim)
        self.dropout = config.attention_dropout

        # these are matrix multiplcation + bias
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Step 1:
        # the input of the forward method is the output of the encoder layer after normalization
        # hidden_states: [Batch_Size, Num_Patches, Embed_Dim], num patches can also be thought of as a sequence length
        batch_size, seq_len, _ = hidden_states.size()
        # query_states: [Batch_Size, Num_Patches, Embed_Dim]
        query_states = self.q_proj(hidden_states)
        # key_states: [Batch_Size, Num_Patches, Embed_Dim]
        key_states = self.k_proj(hidden_states)
        # value_states: [Batch_Size, Num_Patches, Embed_Dim]
        value_states = self.v_proj(hidden_states)
        # query_states: [Batch_Size, Num_Heads, Num_Patches, Head_Dim]
        # Step 2:
        # view splits up the state as above format once the transpose is applied (would be num patches, num heads without the transpose)
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Step 3: 
        # Calculate the attention using the formula Q * K^T / sqrt(d_k). attn_weights: [Batch_Size, Num_Heads, Num_Patches, Num_Patches]
        attn_weights = (torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale)

        if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, seq_len, seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        # Apply the softmax row-wise. attn_weights: [Batch_Size, Num_Heads, Num_Patches, Num_Patches]
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # Apply dropout only during training
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        # Multiply the attention weights by the value states. attn_output: [Batch_Size, Num_Heads, Num_Patches, Head_Dim]
        # Step 4: 
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        # Step 5: transpose back
        # [Batch_Size, Num_Heads, Num_Patches, Head_Dim] -> [Batch_Size, Num_Patches, Num_Heads, Head_Dim]
        attn_output = attn_output.transpose(1, 2).contiguous()  # contigous saves memory
        # Step 6:
        # [Batch_Size, Num_Patches, Num_Heads, Head_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim)
        # [Batch_Size, Num_Patches, Embed_Dim]
        # Step 7: multiply by params
        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights

class SiglipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        # takes each embedding and expands them into intermediate size
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        # shrinks it back into the hidden size

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # [batch size, num patches, embed dim] -> [batch size, num patches, intermediate size]
        hidden_states = self.fc1(hidden_states)
        # non linear transformation
        # hidden states: [batch size, num patches, intermediate size]
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")
        # [batch size, num patches, intermediate size]

        
class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor
    ) -> torch.Tensor:
        # this replicates the encoder as seen in the transformer paper, visually very similar
        # residual: [batch size, num patches, embed data]
        residual = hidden_states
        # [batch size, num patches, embed dim] -> [batch size, num patches, embed dim]
        hidden_states = self.layer_norm1(hidden_states)
        # [batch size, num patches, embed dim] -> [batch size, num patches, embed dim]
        hidden_states, _ = self.self_attn(hidden_states=hidden_states)
        # [batch size, num patches, embed dim]
        hidden_states = residual + hidden_states
        # residual: [batch size, num patches, embed dim]
        residual = hidden_states
        # [batch size, num patches, embed dim] -> [batch size, num patches, embed dim]
        hidden_states = self.layer_norm2(hidden_states)
        # [batch size, num patches, embed dim] -> [batch size, num patches ,embed dim]
        hidden_states = self.mlp(hidden_states)

class SiglipEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        # creating a list of encoder layers which we pass config into
        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    # Ignore copy
    def forward(
        self,
        inputs_embeds: torch.Tensor
    ) -> torch.Tensor:
        # inputs_embeds: [Batch_Size, Num_Patches, Embed_Dim]
        hidden_states = inputs_embeds

        for encoder_layer in self.layers:
            # [Batch_Size, Num_Patches, Embed_Dim] -> [Batch_Size, Num_Patches, Embed_Dim]
            hidden_states = encoder_layer(hidden_states)
            # the output of the last layer becomes the input in the next layer

        return hidden_states


class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.embeddings = SiglipVisionEmbeddings(config)
        # extracts the embeddings from the patches via 2d conv and adds pos enc
        self.encoder = SiglipEncoder(config)
        # then run it through an encoder
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel values: [batch size, channels, height, width] -> [batchsize, num patches, embed dim]
        # converting the patches into embeddings
        hidden_states = self.embeddings(pixel_values)

        last_hidden_state = self.encoder(inputs_embeds=hidden_states)

        last_hidden_state = self.post_layernorm(last_hidden_state)

        return last_hidden_state


class SiglipVisionModel(nn.Module):

    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.config = config  # config defined above
        self.vision_model = SiglipVisionTransformer(config)  # pass config into transformer

    def forward(self, pixel_values) -> Tuple:
        # take the pixel values of our image, loaded with numpy, with the below dimensions
        # the trasnformer converts it into the next dimensions
        # below is [img] -> [embeddings]
        # [batch size, channels, height, width] -> [batchsize, num patches, embed dim]
        return self.vision_model(pixel_values=pixel_values)
    
