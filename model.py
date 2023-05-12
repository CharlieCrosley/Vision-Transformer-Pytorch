from dataclasses import dataclass
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import inspect


@dataclass
class TransformerConfig:
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    device: str = 'cpu'
    image_size: int = 28
    patch_size: int = 6
    total_patches: int = (image_size // patch_size) ** 2
    projection_dim: int = 128
    num_tokens: int = 4
    input_shape: tuple = (1, 1, 28, 28)
    use_token_learner: bool = True


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class PositionalEncoding(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.position_embeddings = nn.Embedding(config.total_patches, config.projection_dim, device=config.device)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, projected_patches, num_patches):
        # Build the positions.
        positions = torch.arange(0, num_patches, dtype=torch.long, device=projected_patches.device)

        # Encode the positions with an Embedding layer.
        encoded_positions = self.position_embeddings(positions)

        # Add encoded positions to the projected patches.
        return self.dropout(projected_patches + encoded_positions)
    


class SelfAttention(nn.Module):

    def __init__(self, config, causal):
        super().__init__()
        # Ensure that the embedding can be split up into n heads
        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.projection_dim = config.projection_dim
        self.causal = causal

        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        # 3 * config.n_embd so that the output can be split into key, query and value tensors.
        # Saves having to make 3 different linear layers
        self.qvk_proj = nn.Linear(config.projection_dim, 3 * config.projection_dim, bias=config.bias)
        # output projection
        self.out_proj = nn.Linear(config.projection_dim, config.projection_dim, bias=config.bias)

    def forward(self, x, enc_in=None, key_padding_mask=None):
        b, n_patches, proj_dim = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.qvk_proj(x).split(self.projection_dim, dim=2)

        if enc_in is not None:
            _, k, v  = self.qvk_proj(enc_in).split(self.projection_dim, dim=2)

        q = q.view(b, n_patches, self.n_head, proj_dim // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        k = k.view(b, n_patches, self.n_head, proj_dim // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(b, n_patches, self.n_head, proj_dim // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.view(b, 1, 1, n_patches).expand(-1, self.n_head, -1, -1).reshape(b * self.n_head, 1, n_patches)
            attn_mask = key_padding_mask.view(b, self.n_head, -1, n_patches)
        else:
            attn_mask = None

        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            # much faster than other implementation!!
            attn_weight = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0, is_causal=self.causal)
        else:
            attn_dropout = self.dropout if self.training else 0
            attn = (q @ k.transpose(-2, -1)) / math.sqrt(q.size(-1))
            # Mask padding tokens
            if attn_mask is not None:
                attn = attn.masked_fill(attn_mask == 0, -float('inf')) # Mask out pad tokens
            # Mask future tokens
            if self.causal:
                attn_mask = torch.ones(n_patches, n_patches, device=x.device).tril(diagonal=0)
                attn = attn.masked_fill(attn_mask == 0, -float('inf')) # Mask out future tokens
            attn_weight = torch.softmax(attn, dim=-1)
            attn_weight = torch.dropout(attn_weight, attn_dropout, self.training) 
            attn_weight  = attn_weight @ v # (b, nh, seq_len, seq_len) x (b, nh, seq_len, hs) -> (b, nh, seq_len, hs)
            
        attn_weight = attn_weight.transpose(1, 2).contiguous().view(b, n_patches, proj_dim) # re-assemble all head outputs side by side
        # output projection
        attn_weight = self.resid_dropout(self.out_proj(attn_weight))
        
        return attn_weight


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.fc    = nn.Linear(config.projection_dim, 4 * config.projection_dim, bias=config.bias)
        self.out_proj  = nn.Linear(4 * config.projection_dim, config.projection_dim, bias=config.bias)
        self.gelu = nn.GELU(approximate='tanh')
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.fc(x)
        x = self.gelu(x)
        x = self.out_proj(x)
        x = self.dropout(x)
        return x
    
class TokenLearner(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config

        h = int(math.sqrt(config.total_patches))
        self.layer_norm_1 = LayerNorm(h, bias=config.bias) #config.projection_dim

        # Applying Conv2d => Reshape => Permute
        # The reshape and permute is done to help with the next steps of
        # multiplication and Global Average Pooling.
        self.attention_maps = nn.Sequential(
                # 3 layers of conv with gelu activation as suggested
                # in the paper.
                nn.Conv2d(
                    in_channels=config.projection_dim,
                    out_channels=config.num_tokens,
                    kernel_size=(3, 3),
                    padding="same",
                    bias=False,
                ),
                nn.GELU(approximate='tanh'),
                nn.Conv2d(
                    in_channels=config.num_tokens,
                    out_channels=config.num_tokens,
                    kernel_size=(3, 3),
                    padding="same",
                    bias=False,
                ),
                nn.GELU(approximate='tanh'),
                nn.Conv2d(
                    in_channels=config.num_tokens,
                    out_channels=config.num_tokens,
                    kernel_size=(3, 3),
                    padding="same",
                    bias=False,
                ),
                nn.GELU(approximate='tanh'),
                # This conv layer will generate the attention maps
                nn.Conv2d(
                    in_channels=config.num_tokens,
                    out_channels=config.num_tokens,
                    kernel_size=(3, 3),
                    padding="same",
                    bias=False,
                ),
                nn.Sigmoid()
        )

        # TODO: Add MLP to learn the attention maps.
        #self.attention_maps = MLP(config)

    def forward(self, inputs):
        # Layer normalize the inputs.
        inputs = self.layer_norm_1(inputs) # (B, H, W, C) # (B, C, H, W)

        attention_maps  = self.attention_maps(inputs)  

        # Reshape and Permute
        attention_maps = attention_maps.view(inputs.shape[0], self.config.num_tokens, -1) # (B, num_of_tokens, H*W)
        
        # Reshape the input to align it with the output of the conv block.
        num_filters = inputs.shape[1]
        inputs = inputs.view((inputs.shape[0], 1, -1, num_filters))  # inputs == (B, 1, H*W, C)

        # Element-Wise multiplication of the attention maps and the inputs
        attended_inputs = torch.mul(attention_maps.unsqueeze(-1), inputs) # (B, num_tokens, H*W, C) 

        # Global average pooling the element wise multiplication result.
        outputs = torch.mean(attended_inputs, axis=2, keepdim=False) # (B, num_tokens, C)
        return outputs

class EncoderBlock(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.multi_head_attn = SelfAttention(config, causal=False)
        self.layer_norm_1 = LayerNorm(config.projection_dim, bias=config.bias)
        self.feed_forward = MLP(config)
        self.layer_norm_2 = LayerNorm(config.projection_dim, bias=config.bias)

    def forward(self, x, key_padding_mask=None):
        x = x + self.multi_head_attn(self.layer_norm_1(x), key_padding_mask=key_padding_mask)
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x

class Encoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([EncoderBlock(config) for _ in range(config.n_layer)])
        self.layer_norm = LayerNorm(config.projection_dim, bias=config.bias)
        self.token_learner = TokenLearner(config)

    def forward(self, x, key_padding_mask=None):
        for i, layer in enumerate(self.layers):
            x = layer(x, key_padding_mask)

            # Add TokenLearner layer in the middle of the
            # architecture. The paper suggests that anywhere
            # between 1/2 or 3/4 will work well.
            if self.config.use_token_learner and i == self.config.n_layer // 2:
                b, hh, c = x.shape
                h = int(math.sqrt(hh)) # h * h = hh
                x = x.view(b, c, h, h) # (B, projection_dim, h, h)
                x = self.token_learner(x) # (B, num_tokens, c)
        return self.layer_norm(x)

class TransformerImageClassifier(nn.Module):

    def __init__(self, config, num_classes):
        super().__init__()

        self.config = config

        num_channels = config.input_shape[1].item()
        num_patches = config.image_size // config.patch_size
        
        self.create_patches = nn.Conv2d(
            in_channels=num_channels,
            out_channels=config.total_patches,
            kernel_size=(num_patches, num_patches),
            stride=(num_patches, num_patches)
        )
        self.project_patches = nn.Linear(config.patch_size * config.patch_size, config.projection_dim, bias=False)

        self.transformer = nn.ModuleDict(dict(
            positional_enc = PositionalEncoding(config),
            encoder = Encoder(config)
        )) 

        self.layer_norm = LayerNorm(config.projection_dim, bias=config.bias)
        self.dense = nn.Linear(config.projection_dim, num_classes, bias=False)

       
        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('out_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))


    def forward(self, inputs):
        data = inputs[0].to(self.config.device)
        labels = inputs[1].to(self.config.device)

        patches = self.create_patches(data) # (B, num_patches, image_size // patch_size, image_size // patch_size)

        b, total_patches, patch_num_h, patch_num_w = patches.shape
        patches = patches.view(b, total_patches, patch_num_h * patch_num_w)  # (B, total_patches, patch_num_h * patch_num_w)
        projected_patches = self.project_patches(patches)  # (B, number_patches, projection_dim)
        
        # Add positional embeddings to the projected patches.
        encoded_patches = self.transformer.positional_enc(projected_patches, self.config.total_patches) # (B, num_tokens, c)
        encoded_patches = self.transformer.encoder(encoded_patches)

        # Layer normalization and Global average pooling.
        representation = self.layer_norm(encoded_patches)
        representation = torch.mean(representation, dim=1)

        if labels is not None:
            # if we are given some desired targets also calculate the loss
            outputs = self.dense(representation)
            logits = nn.functional.softmax(outputs, dim=1)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            outputs = self.dense(representation[:, [-1], :])
            logits = nn.functional.softmax(outputs) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # subtle: 'transformer.wte.weight' and 'lm_head.weight' are tied, so they
        # will appear in the no_decay and decay sets respectively after the above.
        # In addition, because named_parameters() doesn't return duplicates, it
        # will only return the first occurence, key'd by 'transformer.wte.weight', below.
        # so let's manually remove 'lm_head.weight' from decay set. This will include
        # this tensor into optimization via transformer.wte.weight only, and not decayed.
        #decay.remove('dense.weight')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        # new PyTorch nightly has a new 'fused' option for AdamW that is much faster
        use_fused = (device_type == 'cuda') and ('fused' in inspect.signature(torch.optim.AdamW).parameters)
        print(f"using fused AdamW: {use_fused}")
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

        return optimizer
    
    
    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.positional_enc.position_embeddings.weight.numel()
        return n_params
