out_dir = 'out-mnist'
eval_interval = 1000 # keep frequent because we'll overfit
eval_iters = 250
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = True

wandb_log = False # override via command line if you like
wandb_project = 'mnist'
wandb_run_name = 'mnist-run'

dataset = 'mnist'
gradient_accumulation_steps = 1
batch_size = 32 #254

learning_rate = 1e-4 # with baby networks can afford to go a bit higher
max_iters = 20000
lr_decay_iters = 20000 # make equal to max_iters usually
min_lr = 1e-6 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

device = 'cuda'
# on macbook also add
# device = 'cpu'  # run on cpu only
compile = False # do not torch compile the model

# transformer params
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.1

# AUGMENTATION
patch_size = 4  # Size of the patches to be extracted from the input images.
projection_dim = 156
use_token_learner=True
# TOKENLEARNER
num_tokens = 4