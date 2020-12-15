# -*- coding: utf-8 -*-
from __future__ import absolute_import
from datetime import datetime

import logging
from logging.handlers import RotatingFileHandler
logger = logging.getLogger()

def attach_time_to_file_name(file_name):
    now = datetime.now()
    current_time = now.strftime("_%m%d_%H%M")
    temp_names = file_name.split(".")
    file_name = ".".join(temp_names[:-1])+current_time+"."+temp_names[-1]
    return file_name


def init_logger(log_file=None, log_file_level=logging.NOTSET, rotate=False):
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    log_file = attach_time_to_file_name(log_file)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != '':
        if rotate:
            file_handler = RotatingFileHandler(
                log_file, maxBytes=1000000, backupCount=10)
        else:
            file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger
