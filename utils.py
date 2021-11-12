import pickle
from datetime import datetime


def save_object(object, name):
    file_handler = open(name+datetime.now().strftime("%m_%d_%Y-%H_%M_%S"), 'wb')
    pickle.dump(object, file_handler)


def load_object(name):
    file_handler = open(name, 'rb')
    return pickle.load(file_handler)


class ProgressBar:
    def __init__(self, message, max_value, interval=1):
        self.max_value = max_value
        self.last_prog = 0
        self.interval = interval
        self.number_of_prints = 0
        print('\n'+message+':')
    def print_progress(self, current):
        prog = int((100*current)/self.max_value)
        if prog >= self.last_prog + self.interval:
            self.last_prog = prog
            print('==>'+str(self.last_prog)+'%', end='')
            self.number_of_prints = self.number_of_prints + 1
            if self.number_of_prints % 10 == 0:
                print('\n', end='')


