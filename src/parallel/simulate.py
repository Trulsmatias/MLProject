import logging
import mmap
import multiprocessing as mp
import os
from generations import Generation
from parallel.serialize import sharedmem_individual_to_task, pickle_individual_to_task
from parallel.worker import worker_proc


class ParallelSimulator:
    def __init__(self, simulator, num_workers):
        """
        Creates a ParallelSimulator by wrapping a Simulator.
        The ParallelSimulator distributes simulation of individuals to worker processes.
        :param simulator: the Simulator
        :param num_workers: number of worker processes to use
        """
        self._simulator = simulator
        self._worker_procs = []
        self._workers_running = False
        self._task_queue = mp.JoinableQueue()
        self._result_queue = mp.Queue()
        self._shutdown_workers_event = mp.Event()
        self._log = logging.getLogger('MLProject.parallel.Master')

        simulator_params = {'movements': simulator.movements, 'max_steps': simulator.max_steps}
        self._log.info('Creating worker processes')
        for wnum in range(1, num_workers + 1):
            worker = mp.Process(target=worker_proc,
                                args=(wnum, self._task_queue, self._result_queue,
                                      self._shutdown_workers_event, simulator_params),
                                name='Worker_{}'.format(wnum))
            worker.daemon = True
            self._worker_procs.append(worker)

        self._log.info('Creating memory-mapped file')
        # self._mmap_fd = open('mmap.dat', 'w+b')
        # self._mmap_fd.write(b'\x00' * 1024 * 1024 * 100)
        # self._mmap = mmap.mmap(self._mmap_fd.fileno(), 0)

    def _spawn_workers(self):
        if not self._workers_running:
            self._log.info('Spawning worker processes')
            for worker in self._worker_procs:
                worker.start()
            self._workers_running = True

    def simulate_generation(self, generation: Generation, render=False):
        """
        Enqueues simulation of the generation to worker processes, which simulates the
        individuals and assigns each a fitness score.
        :param generation:
        :param render: always False, has no effect
        """
        # Ensure worker processes are started
        if not self._workers_running:
            self._spawn_workers()

        self._log.debug('Filling task queue')
        for individual in generation.individuals:
            task = pickle_individual_to_task(individual)
            self._task_queue.put(task)
        self._log.debug('Task queue filled with {} elements'.format(len(generation.individuals)))

        # Wait until workers have processed all tasks
        self._log.debug('Waiting until workers have processed all tasks')
        self._task_queue.join()

        # Iterate through result queue and set fitnesses to individuals
        self._log.debug('Setting fitnesses to individuals from worker results')
        while not self._result_queue.empty():
            result = self._result_queue.get()
            individual = next(i for i in generation.individuals if i.id == result.id)
            if not individual:
                self._log.error('Invalid individual in worker result: {}'.format(result.id))
            else:
                individual.fitness = result.fitness

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

        # self._mmap.close()
        # self._mmap_fd.close()
