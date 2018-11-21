import logging
import signal
import sys
import multiprocessing as mp
import os
import queue
from neat.config import setup_logging


def worker_proc(worker_num, task_queue: mp.JoinableQueue, result_queue: mp.Queue,
                shutdown_event: mp.Event, simulator_params):

    setup_logging(simulator_params['log_level'])
    log = logging.getLogger('NEAT.parallel.Worker_{}'.format(worker_num))

    def shutdown(sig=0, sframe=None):
        log.info('Worker shutting down')
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    from neat import SimulateMario
    log.info('Worker started. PID: {}. Parent PID: {}'.format(os.getpid(), os.getppid()))

    while True:
        try:
            log.debug('Checking for tasks')
            genome = task_queue.get(timeout=1)

            log.info('Simulating genome {}-{}'.format(genome.gen_num, genome.id))
            SimulateMario.simulate_run(genome, simulator_params['max_steps'], simulator_params['render'])

            result_queue.put({'id': genome.id, 'fitness': genome.fitness})
            task_queue.task_done()

        except queue.Empty:
            log.debug('Task queue is empty')

        if shutdown_event.is_set():
            shutdown()
            return
