from typing import NamedTuple


class ModelConfig(NamedTuple):
    vocab_size = 7244
    embed_size = 128
    encoder_size = 128
    max_context_len = 10
    min_cnt = 5
    encoder_type = 'gru'
    is_temp_enc = True


class TrainConfig(NamedTuple):
    """ Hyperparameters for training """
    seed = 123  # random seed
    batch_size = 32
    lr = 0.001  # learning rate
    n_epochs = 10  # the number of epoch
    # `warm up` period = warmup(0.1)*total_steps
    # linearly increasing learning rate from zero to the specified value(5e-5)
    warmup = 0.1
    save_steps = 1000  # interval for saving persona_model
    total_steps = 100000  # total number of steps to train
