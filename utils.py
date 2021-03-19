import os
from datetime import datetime


class Logger(object):

    def __init__(self, verbose=0, log_path=None, file_prefix=""):
        self.verbose = verbose
        self.filename = None
        if log_path is not None:
            if not os.path.exists(log_path):
                os.makedirs(log_path)
            self.filename = os.path.join(
                log_path, file_prefix + ".log")
            with open(self.filename, "w") as f:
                f.write(self.filename)
                f.write("\n")

    def p(self, s, level=1):
        if self.verbose >= level:
            print(s)
        if self.filename is not None:
            with open(self.filename, "a") as f:
                f.write(datetime.now().strftime("[%m/%d %H:%M:%S]  ") + str(s))
                f.write("\n")
