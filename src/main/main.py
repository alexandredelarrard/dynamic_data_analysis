# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 16:03:38 2017

@author: alexandre
"""

from multiprocessing import Queue
from client_utils.multithreading_gpu_cpu import multithreading_cpu
 

class MainClient(object):
    def __init__(self, task):

        cpu_queue, task_cpu_queue, completed_queue = self.get_server_ressources()
        self.tasks_table = {}
        
        self.main(cpu_queue, task_cpu_queue, completed_queue, task)

    def main(self, cpu_queue, task_cpu_queue, completed_queue, task):
        
         current_task_id = ""
         while True:
             
            if task["id"] != current_task_id:
                
                current_task_id = task["id"]
                
                # initialise the task_table
                self.execute_task(task, task_cpu_queue)
                self.put_available_task(task["id"], task_cpu_queue)

            if not completed_queue.empty():
                task_i = completed_queue.get()
                self.tasks_table[task_i["id_main_task"]]["sub_task"][task_i["id_sub_task"]]["status"]  = task_i["status"]
                self.tasks_table[task_i["id_main_task"]]["sub_task"][task_i["id_sub_task"]]["message"] = task_i["message"]
                self.tasks_table[task_i["id_main_task"]]["sub_task"][task_i["id_sub_task"]]["output"]  = task_i["output"]
                self.tasks_table[task_i["id_main_task"]]["desc_task"]["nbr_remaining"] -=1
                
                if task_i["status"] == "error":
                    for ids_node in self.tasks_table[task_i["id_main_task"]]["sub_task"][task_i["id_sub_task"]]["sons"]:
                        del self.tasks_table[task_i["id_main_task"]]["sub_task"][ids_node]
                        self.tasks_table[task_i["id_main_task"]]["desc_task"]["nbr_remaining"] -=1
                    # have to send message of error here
                    print(task_i["message"])
                    del self.tasks_table[task_i["id_main_task"]]["sub_task"][task_i["id_sub_task"]]
                    
                #### if all sub tasks completed, then the last update has the update results table ---> send to s3
                if self.tasks_table[task_i["id_main_task"]]["description_tasks"]["nbr_remaining"] == 0 :
                    del self.tasks_table[task_i["id_main_task"]]
                    # send results giving info if all process has been done or an error occured
                    
                else:
                    self.put_available_task(task_i["id_main_task"], task_cpu_queue) 
                    
                                        
    def execute_task(self, task, task_cpu_queue):

        ##### Set class variables for sub classes that inherit from main_client
        sons = {} 
        self.tasks_table[task["id_main_task"]] = {"sub_task"         : {},
                                                  "desc_task" : {"execution_time": 0, 
                                                                       "nbr_remaining": len(task["sub_task"])},
                                                  "company"   : task["company"],
                                                  "account"   : task["account"]}
                                               
        for key, value in task["sub_task"].iteritems():
            
            if key not in sons.keys():
                sons[key] = []
            for sub_task_ids in task["sub_task"][key]["parents"]:
                sons[sub_task_ids] += [key]
                
            self.tasks_table[task["id_main_task"]]["sub_task"][key] = { "id_main_task" : task["id_main_task"],
                                                                        "id_sub_task" : key,
                                                                        "input" : task["sub_task"][key]["input"],
                                                                        "output" : {},
                                                                        "mode"   : task["sub_task"][key]["mode"],
                                                                        "module" : task["sub_task"][key]["module"],
                                                                        "method" : task["sub_task"][key]["method"],
                                                                        "model"  : task["sub_task"][key]["model"],
                                                                        "parameters": task["sub_task"][key]["parameters"],
                                                                        "gpu_cpu"  : "cpu",
                                                                        "exec_time":  0,
                                                                        "start_time": 0,
                                                                        "parents": task["sub_task"][key]["parents"],
                                                                        "sons"   : [],
                                                                        "status" : "waiting" if len(task["sub_task"][key]["parents"]) != 0  else "pending",
                                                                        "message": ""}
        
        for key, values in sons.iteritems():
            self.tasks_table[task["id_main_task"]]["sub_task"][key]["sons"] = values
        
        task_cpu_queue.put(self.tasks_table[task["id_main_task"]]["sub_task"][key])
        
                      
    def put_available_task(self, id_task, task_cpu_queue):

        sub_table = self.tasks_table[id_task]
        
        for key in sub_table.keys():
            available = True
            if len(sub_table[key]["parents"])>0:
                for parents_id in sub_table[key]["parents"]:
                    if sub_table[parents_id]["status"] != "completed":
                        available = False
                        
            if available == True:
                self.tasks_table[id_task]["sub_task"][key]["status"] = "pending"
                for x in self.tasks_table[id_task]["sub_task"][key]["parents"]:
                    self.tasks_table[id_task]["sub_task"][key]["input"]  = self.tasks_table[id_task]["sub_task"][key]["input"].update(self.tasks_table[id_task]["sub_task"][x]["output"])
                
                task_cpu_queue.put(self.tasks_table[id_task]["sub_task"][key])
        
        
    def get_server_ressources(self):
        cpu_queue = Queue()
        
        for i in range(2):
            cpu_queue.put(i)

        task_cpu_queue = Queue()
        completed_queue= Queue()

        multithreading_cpu(task_cpu_queue, cpu_queue, completed_queue)
        
        return cpu_queue, task_cpu_queue, completed_queue
    

if __name__ == "__main__":
    
    task = {}
    MainClient(task)
 