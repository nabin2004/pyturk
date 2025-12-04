
class Logger:
    def __init__(self):
        self.logs = {}

    def log(self, name, value):
        if name not in self.logs:
            self.logs[name] = []
        self.logs[name].append(value)

    def summary(self, name):
        vals = self.logs.get(name, [])
        if vals:
            return {
                'mean': sum(vals)/len(vals),
                'max': max(vals),
                'min': min(vals)
            }
        return {}

    def reset(self):
        self.logs = {}
