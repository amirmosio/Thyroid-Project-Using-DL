import os


def set_config_for_logger(config_label):
    import logging
    trains_state_dir = "./train_state"
    if not os.path.isdir(trains_state_dir):
        os.mkdir(trains_state_dir)
    config_train_dir = os.path.join(trains_state_dir, config_label)
    if not os.path.isdir(config_train_dir):
        os.mkdir(config_train_dir)
    log_file = os.path.join(config_train_dir, "console.log")
    logger = logging.getLogger(config_label)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s|%(levelname)s|%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    return logger
