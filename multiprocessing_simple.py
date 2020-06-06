# -*- coding: utf-8 -*-
"""
multiprocessing_simple.py

"""

import multiprocessing

def worker(num):
    """Returns the string of interest"""
    return "worker %d" % num

def worker_mp(n):
    pool = multiprocessing.Pool()
    results = pool.map(worker, range(n))

    pool.close()
    pool.join()

    for result in results:
        # prints the result string in the main process
        print(result)

if __name__ == '__main__':
    # Better protect your main function when you use multiprocessing
    worker_mp(10)