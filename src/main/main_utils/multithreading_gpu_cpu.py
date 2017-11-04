# -*- coding: utf-8 -*-
"""
Created on Thu Jun 01 12:08:45 2017

@author: alexandre
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jun 01 12:08:45 2017

@author: alexandre
"""

import os, sys
from multiprocessing import Process
import traceback
import signal
import time
from datetime import datetime
import logging

from client_utils.check_data          import check_client_data
from client_utils.check_models        import check_client_models

sys.path.append("/".join([os.environ["Q_PATH"], "src"]))
from crawl.main_crawl                           import Main_crawl
from mdl.main_mdl                               import Main_mdl
from optim.main_optim                           import Main_optim
from visu.main_visu                             import Main_visu
from data_management.main_data_management       import Main_data_management


def multithreading_cpu(task_cpu_queue, cpu_queue, completed_queue):

    jobs = []
    for process_id in range(cpu_queue.qsize()):
        p = Process(name='cpu_process', target= launch_threads, args=(task_cpu_queue, cpu_queue, completed_queue))
        jobs.append(p)
        p.daemon =True
        p.start()


def launch_threads(task_queue, pu_queue, completed_queue):

    while True:
        if not pu_queue.empty() and not task_queue.empty():

            task_i = task_queue.get()
            pu_id = pu_queue.get()

            try:
                logging.error("pid is %s; type = (%s, %s, %s)"%(os.getpid(), task_i["module"], task_i["method"], task_i["model"]))
                
                start = time.time()
                class_model = globals()["Main_%s"%task_i["module"]](task_i)               
                
                if task_i["mode"] == "train":
                    print("Start train")
                    task_i  = class_model.Train()
                    print("End train")
                
                elif task_i["mode"] == "test":
                    print("Start train")
                    task_i  = class_model.Test()
                    print("End train")
                
                elif task_i["mode"] == "prod":
                    print("Start train")
                    task_i  = class_model.Prod()
                    print("End train")
                
                else :       
                    print(" The mode must be either train, test or prod")
                    
                task_i["date"]   = datetime.now().strftime('%Y/%m/%d %H:%M:%S')                            
                task_i["time"]   = time.time() - start
                
                ### free cpu queue and add finished task to completed queue
                completed_queue.put(task_i)                    
                pu_queue.put(pu_id)

            except Exception as e:
                var = traceback.format_exc()
                logging.error(var)
                traceback.print_exc(file=sys.stderr)
                os.kill(os.getpid(), signal.SIGTERM)
                task_i["message"] = e
                task_i["status"]  = "error"
                completed_queue.put(task_i)
                pu_queue.put(pu_id)
                pass

                