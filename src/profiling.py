import os
import psutil
import logging
import objgraph

_log = logging.getLogger('Profiler')


def sizeof_fmt(num, suffix='B'):
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


def start():
    global _ps
    _log.debug('PID: {}'.format(os.getpid()))
    _ps = psutil.Process(os.getpid())


def mem():
    _log.debug('Memory usage: {}'.format(sizeof_fmt(_ps.memory_info().rss)))


def obj():
    _log.debug('Generations: {}'.format(len(objgraph.by_type('Generation'))))
    _log.debug('Individuals: {}'.format(len(objgraph.by_type('Individual'))))
    _log.debug('NNAgents: {}'.format(len(objgraph.by_type('NNAgent'))))
    _log.debug('Sequentials: {}'.format(len(objgraph.by_type('Sequential'))))

    for i, obj in enumerate(objgraph.by_type('Sequential')):
        # objgraph.show_chain(objgraph.find_backref_chain(obj, objgraph.is_proper_module),
        #                    filename='refs/chain_{}.png'.format(i))
        # objgraph.show_backrefs(obj, max_depth=5, too_many=5, filename='refs/backrefs_{}.png'.format(i))
        pass

    # objgraph.show_refs(current_generation, filename='refs.png')
