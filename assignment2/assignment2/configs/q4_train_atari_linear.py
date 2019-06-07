class config():
    # env config
    render_train     = True
    render_test      = False
    env_name         = "Pong-v0"
    overwrite_render = True
    record           = True
    high             = 255.

    # output config
    output_path  = "results/q4_train_atari_linear/"
    model_output = output_path + "model.weights/"
    log_path     = output_path + "log.txt"
    plot_output  = output_path + "scores.png"
    record_path  = output_path + "monitor/"

    # model and training config
    num_episodes_test = 50
    grad_clip         = True
    clip_val          = 10
    saving_freq       = 250000
    log_freq          = 50
    eval_freq         = 25000
    record_freq       = 2500
    soft_epsilon      = 0.05

    # nature paper hyper params
    nsteps_train       = 5000000
    batch_size         = 100
    buffer_size        = 1000000
    target_update_freq = 1000
    gamma              = 1.00
    learning_freq      = 10
    state_history      = 10
    skip_frame         = 4
    lr_begin           = 0.001
    lr_end             = 0.0001
    lr_nsteps          = nsteps_train
    eps_begin          = 1
    eps_end            = 0.05
    eps_nsteps         = 10000000
    learning_start     = 1000
