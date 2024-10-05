class Hyperparams(dict):
    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, attr, value):
        self[attr] = value


hps = Hyperparams(
    seq_len=64000,
    down_t=7,  # 8192 / 2**7 = 64
    stride_t=2,  # Base of the above power
    emb_width=64,
    l_bins=2048,  # Codebook size
    l_mu=0.99,
    commit=0.02,
    spectral=0.0,
    multispectral=1.0,
    hvqvae_multiplier=1,  # increases model size
    loss_fn='l2',
    linf_k=2048,
    use_bottleneck=True,
    revival_threshold=1.0,
    samples_per_inst=50,

    # Audio
    bandwidth={'l1': 1., 'l2': 1., 'spec': 1.},
    n_fft=1024,
    hop_length=256,
    window_size=1024,
    sr=16000,
    channels=1,
    wav='',
    n_inps=1,
    n_hops=2,
    n_segment=1,
    n_total_segment=1,
    n_segment_each=1,
    prime_chunks=4,
    sample_hop_length=30000,
    max_silence_pad_length=0,
    ignore_boundaries=False,

    multispec_loss_n_fft=(2048,1024,512),
    multispec_loss_hop_length=(240,120,50),
    multispec_loss_window_size=(1200,600,240),
)

hps_tiny = Hyperparams(
    down_t=3,  # power
    stride_t=4,  # Base of the above power
    emb_width=1,  # original 64
    hvqvae_multiplier=1,
    samples_per_inst = 2,
    l_bins=16,  # Codebook size. original 2048
)

hps_opt = Hyperparams(
    epochs=10000,
    lr=0.0003,
    clip=1.0,
    beta1=0.9,
    beta2=0.999,
    ignore_grad_norm=0,
    gn_scale = 1.0,
    weight_decay=0.0,
    eps=1e-08,
    lr_warmup=100.0,  # lr starts at 0 and increases to lr for lr_warmup steps
    lr_decay=10000000000.0,
    lr_gamma=1.0,
    lr_scale=1.0,
    lr_use_linear_decay=False,
    lr_start_linear_decay=0,
    lr_use_cosine_decay=False,

    fp16=False,
    fp16_params=False,
    fp16_loss_scale=None,
    fp16_scale_window=1000.0,
    fp16_opt=False,

    prior=False,
    restore_vqvae='',
    mu=0.99,
    bs=1,
    ngpus=0,
    ema=True,
    train=True,
)

block_kwargs = Hyperparams(
    depth=4,
    width=32,
    m_conv=1.0,
    dilation_growth_rate=3,
    dilation_cycle=None,
)
