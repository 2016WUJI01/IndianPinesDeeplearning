import logging

def get_logger(filename, verbosity=1, name=None):
    lever_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING, 3: logging.ERROR}
    formatter = logging.Formatter(
                "[%(asctime)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(lever_dict[verbosity])

    fh = logging.FileHandler(filename, "a+")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger