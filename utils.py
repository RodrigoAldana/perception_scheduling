import pickle
from datetime import datetime


def save_object(object, name):
    file_handler = open(name+datetime.now().strftime("%m_%d_%Y-%H_%M_%S"), 'wb')
    pickle.dump(object, file_handler)


def load_object(name):
    file_handler = open(name, 'rb')
    return pickle.load(file_handler)


class ProgressBar:
    def __init__(self, message, max_value, interval=1, silent=False):
        self.max_value = max_value
        self.last_prog = 0
        self.interval = interval
        self.number_of_prints = 0
        self.silent = silent
        if not silent:
            print('\n'+message+':')

    def print_progress(self, current, message=''):
        prog = (100.0*current)/float(self.max_value)
        if prog >= self.last_prog + self.interval:
            self.last_prog = prog
            if not self.silent:
                print(message+'==>'+('%.1f' % self.last_prog)+'%', end='')
            self.number_of_prints = self.number_of_prints + 1
            if self.number_of_prints % 10 == 0 and not self.silent:
                print('\n', end='')
        if current >= self.max_value-1 and not self.silent:
            print(message + '==>' + '100%')


