import threading
class ThreadWithResult(threading.Thread):
    def __init__(self, target, args=()):
        super(ThreadWithResult, self).__init__()
        self.target = target
        self.args = args
        self.result = None

    def run(self):
        self.result = self.target(*self.args)