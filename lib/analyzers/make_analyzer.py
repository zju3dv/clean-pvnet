import os
import imp


def make_analyzer(cfg):
    module = '.'.join(['lib.analyzers', cfg.task])
    path = os.path.join('lib/analyzers', cfg.task+'.py')
    analyzer = imp.load_source(module, path).Analyzer()
    return analyzer
