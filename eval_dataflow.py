# this file is prepared for project 511
# Created by iboxl

import itertools
from Architecture.ArchSpec import CIM_Acc
from utils.Workload import WorkLoad
from utils.Tools import *
import gurobipy as gp
from gurobipy import GRB, quicksum
from utils.SolverSP import Solver as Solver
from Simulator.Calculator import calc_ds
from utils.GlobalUT import *
import numpy as np
from Architecture.Acc_1 import accelerator as acc_zz
import pickle
from Simulator.Simulax import tranSimulator
   

if __name__ == "__main__":
    Logger.setcfg(setcritical=False, setDebug=False, STD=False, file='511.log', nofile=False)

    import logging
    CONST.FLAG_OPT="Latency"
    Logger.setLevel(logging.DEBUG)
    cfg = get_ConfigFile('cim_template.cfg')

    with open(f"Dataflow.pkl", 'rb') as fp:
        dataflow = pickle.load(fp)
    
    Logger.info(f'Evaluation in simulator:') 

    acc = CIM_Acc(acc_zz.cores[0])
    cfg = get_ConfigFile('cim_template.cfg')
    # ops = WorkLoad(cfg, loopDim={'R': 3, 'S': 3, 'C': 128, 'K':128, 'P': 56, 'Q': 56, 'G': 1, 'B': 1, 'H': 56, 'W': 56, 'Stride': 1, 'Padding': 1},
    # ops = WorkLoad(cfg, loopDim={'R': 1, 'S': 1, 'C': 128, 'K':256, 'P': 14, 'Q': 14, 'G': 1, 'B': 1, 'H': 28, 'W': 28, 'Stride': 2, 'Padding': 0},
    # ops = WorkLoad(cfg, loopDim={'R': 3, 'S': 3, 'C': 128, 'K':128, 'P': 28, 'Q': 28, 'G': 1, 'B': 1, 'H': 28, 'W': 28, 'Stride': 1, 'Padding': 1},
    ops = WorkLoad(cfg, loopDim={'R': 1, 'S': 1, 'C': 64, 'K':128, 'P': 28, 'Q': 28, 'G': 1, 'B': 1, 'H': 56, 'W': 56, 'Stride': 2, 'Padding': 0},
                   )

    # Logger.info(f'R:{ops.R}, S:{ops.S}, C:{ops.C}, K:{ops.K}, P:{ops.P}, Q:{ops.Q}, H:{ops.H}, W:{ops.W}, G:{ops.G}, B:{ops.B}, Stride:{ops.Stride}, Padding:{ops.Padding}')

    #  ['Dram'1, 'Global_buffer'2, 'Output_buffer'3, 'Input_buffer'4, 'OReg'5, 'IReg'6, 'Macro'7]
    simu = tranSimulator(acc=acc, ops=ops, dataflow=dataflow)

    l_zz, e_zz = simu.run()

    simu.LModeling()
    Logger.info(f"* * * Zigzag-Running  * * *  Latency:{round(l_zz,3):<15}, Energy:{round(e_zz,3):<15}, EDP:{round(l_zz *e_zz,3):.5e}")
