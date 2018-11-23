import logging
import multiprocessing as mp
import queue
from neat.parallel.worker import worker_proc


class ParallelSimulator:
    def __init__(self, num_workers, max_steps, render):
        """
        Creates a ParallelSimulator.
        The ParallelSimulator distributes simulation of genomes to worker processes.
        """
        self._worker_procs = []
        self._workers_running = False
        self._task_queue = mp.JoinableQueue()
        self._result_queue = mp.Queue()
        self._shutdown_workers_event = mp.Event()
        self._log = logging.getLogger('NEAT.parallel.Master')

        simulator_params = {'max_steps': max_steps,
                            'render': render,
                            'log_level': self._log.getEffectiveLevel()}
        self._log.info('Creating worker processes')
        for wnum in range(1, num_workers + 1):
            worker = mp.Process(target=worker_proc,
                                args=(wnum, self._task_queue, self._result_queue,
                                      self._shutdown_workers_event, simulator_params),
                                name='Worker_{}'.format(wnum))
            worker.daemon = True
            self._worker_procs.append(worker)

    def _spawn_workers(self):
        if not self._workers_running:
            self._log.info('Spawning worker processes')
            for worker in self._worker_procs:
                worker.start()
            self._workers_running = True

    def simulate_genomes(self, genomes_list):
        """
        Enqueues simulation of the genomes to worker processes, which simulates the
        genomes and assigns each a fitness score.
        """
        # Ensure worker processes are started
        if not self._workers_running:
            self._spawn_workers()

        self._log.debug('Filling task queue')
        for genome in genomes_list:
            self._task_queue.put(genome)
        self._log.debug('Task queue filled with {} elements'.format(len(genomes_list)))

        # Wait until workers have processed all tasks
        self._log.debug('Waiting until workers have processed all tasks')
        self._task_queue.join()

        self._log.debug('Retrieving genome fitnesses from workers')
        processed_genomes = 0
        while processed_genomes < len(genomes_list):
            try:
                result = self._result_queue.get(timeout=30)
                genome = next(g for g in genomes_list if g.id == result['id'])
                if not genome:
                    self._log.error('Invalid genome in worker result: {}'.format(result['id']))
                else:
                    genome.fitness = result['fitness']
                processed_genomes += 1

            except queue.Empty:
                self._log.warning('Retrieving simulation result took too long. Has a worker died?')

    def shutdown(self):
        """
        Shuts down all workers.
        """
        # Send shutdown signal to workers
        self._shutdown_workers_event.set()
        # Wait for all workers to shut down
        self._log.info('Waiting for workers to shut down')
        for worker in self._worker_procs:
            worker.join()
