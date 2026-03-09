#  this file is prepared for project 419
#  Created by iboxl
#  Modified in project 112

from utils.Workload import WorkLoad, LoopNest
from Architecture.ArchSpec import CIM_Acc
import math
from utils.GlobalUT import *
from dataclasses import dataclass, field

@dataclass
class TransCost():
    r:float
    w:float
    t:int       

@dataclass
class ProfilingDetail():
    dynamic_power:float = 0
    leakage_power:float = 0
    dynamic_power_onChip:float = 0
    memCost:list[float] = field(default_factory=list)
    macEnergy:float = 0
    macLatency:int = 0
    latency:int = 0
    Energy_OnChip:float = 0
    Energy_OffChip:float = 0
    doubleFlag:list[list[bool]] = field(default_factory=list)

class tranSimulator():
    def __init__(self, acc:CIM_Acc, ops:WorkLoad, dataflow:LoopNest, DEBUG_SIMU=False):
        Logger.info('* - '*8 + "Deploying Dataflow Onto Trans-simulator" + ' - *'*8)
        self.acc = acc
        self.ops = ops
        self.dataflow = dataflow
        self.DEBUG_SIMU = DEBUG_SIMU

        # from utils.CompatibleZigzag import fix_all_memHierarchy
        # self.dataflow.tm = fix_all_memHierarchy(acc=acc, tm=dataflow.tm)
        self.dataflow.preprogress()
        # print(self.dataflow)
        Logger.info(self.dataflow)

        # dim_tp[mem][t][dim]   
        dim_tp = [ [ [ 1 for dim in range(ops.Num_dim)] for t in range(3) ] for mem in range(acc.Num_mem)]
        # dim_sp[mem][t][dim]
        dim_sp = [ [ [ 1 for dim in range(ops.Num_dim)] for t in range(3) ] for mem in range(acc.Num_mem)]
        for mem in range(1,acc.Num_mem):
            for op, op_name in enumerate(['I','W','O']):
                for mapping in dataflow.tm:
                    if mem <= mapping.mem[op] and acc.mappingArray[op][mem]==1:
                        dim_tp[mem][op][mapping.dim] *= mapping.dimSize
                for mapping in dataflow.sm:
                    if mem <= mapping.mem[op] and acc.mappingArray[op][mem]==1:
                        dim_sp[mem][op][mapping.dim] *= mapping.dimSize
        self.dim_tp, self.dim_sp = dim_tp, dim_sp
        
        dataSize = {}
        tmp_dim = [_ for _ in range(ops.Num_dim)]
        for op, op_name in enumerate(['I','W','O']):
            dataSize[acc.Num_mem, op] = 1
        for mem in range(1,acc.Num_mem):
            op = 0  # Input
            for dim in range(1,ops.Num_dim):
                tmp_dim[dim] = dim_sp[mem][op][dim] * dim_tp[mem][op][dim]
            if acc.mappingArray[op][mem]:
                # dataSize[mem,op] = (tmp_dim[ops.dict2Dim('P')]+tmp_dim[ops.dict2Dim('R')]-1) * (tmp_dim[ops.dict2Dim('Q')]+tmp_dim[ops.dict2Dim('S')]-1) * tmp_dim[ops.dict2Dim('C')]
                # tmp_h = (tmp_dim[ops.dict2Dim('P')] - 1) * ops.Stride + tmp_dim[ops.dict2Dim('R')] # (- 2 * ops.pandding)
                # tmp_w = (tmp_dim[ops.dict2Dim('Q')] - 1) * ops.Stride + tmp_dim[ops.dict2Dim('S')]

                if ops.Stride >= tmp_dim[ops.dict2Dim('R')]:
                    tmp_h = tmp_dim[ops.dict2Dim('P')] * tmp_dim[ops.dict2Dim('R')]
                else:
                    tmp_h = (tmp_dim[ops.dict2Dim('P')] - 1) * ops.Stride + tmp_dim[ops.dict2Dim('R')]
                if ops.Stride >= tmp_dim[ops.dict2Dim('S')]:
                    tmp_w = tmp_dim[ops.dict2Dim('Q')] * tmp_dim[ops.dict2Dim('S')]
                else:
                    tmp_w = (tmp_dim[ops.dict2Dim('Q')] - 1) * ops.Stride + tmp_dim[ops.dict2Dim('S')]

                dataSize[mem,op] = min(tmp_h,ops.H) * min(tmp_w,ops.W) * tmp_dim[ops.dict2Dim('C')] 
            else:
                dataSize[mem,op] = 0

            op = 1  # Weight
            for dim in range(1,ops.Num_dim):
                tmp_dim[dim] = dim_sp[mem][op][dim] * dim_tp[mem][op][dim]
            if acc.mappingArray[op][mem]:
                dataSize[mem,op] = tmp_dim[ops.dict2Dim('R')] * tmp_dim[ops.dict2Dim('S')] * tmp_dim[ops.dict2Dim('C')] * tmp_dim[ops.dict2Dim('K')] 
            else:
                dataSize[mem,op] = 0

            op = 2  # Output
            for dim in range(1,ops.Num_dim):
                tmp_dim[dim] = dim_sp[mem][op][dim] * dim_tp[mem][op][dim]
            if acc.mappingArray[op][mem]:
                dataSize[mem,op] = tmp_dim[ops.dict2Dim('P')] * tmp_dim[ops.dict2Dim('Q')] * tmp_dim[ops.dict2Dim('K')] 
            else:
                dataSize[mem,op] = 0
        for op, op_name in enumerate(['I','W','O']):
            for mem in range(1,acc.Num_mem):
                if self.dataflow.bypassMem[mem][op]:
                    dataSize[mem,op] = 0
        
        Logger.info(f"Utilization:") 
        for mem in range(1,acc.Num_mem):
            sum_mem = 0
            for op, op_name in enumerate(['I','W','O']):
                if acc.mappingArray[op][mem]:
                    sum_mem += (dataSize[mem,op] + dataSize[mem,op] * dataflow.usr_defined_double_flag[mem][op]) * acc.precision[mem,op]
            if sum_mem > acc.memSize[mem]:
                Logger.error(f"OverSize Error:\n{acc.mem2dict(mem)}:{acc.memSize[mem]}(Size) less than {sum_mem}(Used):"
                                + f"I-{dataSize[mem,0]*acc.precision[mem,0]}({dataflow.usr_defined_double_flag[mem][op]}) "
                                + f"W-{dataSize[mem,1]*acc.precision[mem,1]}({dataflow.usr_defined_double_flag[mem][op]}) "
                                + f"O-{dataSize[mem,2]*acc.precision[mem,2]}({dataflow.usr_defined_double_flag[mem][op]})")
                raise ValueError("Dataflow Over MemSize Error")
            else:
                Logger.info('-'*4 + f" {acc.mem2dict(mem)}({round(sum_mem/acc.memSize[mem]*100,2):>3}%): {sum_mem}/{acc.memSize[mem]} - {acc.bw[mem]} bit/cc")
        self.dataSize = dataSize

        # dim_tp[mem][t][dim]   
        trans_dim_tp = [ [ [ 1 for dim in range(ops.Num_dim)] for t in range(3) ] for mem in range(acc.Num_mem)]
        # dim_sp[mem][t][dim]
        trans_dim_sp = [ [ [ 1 for dim in range(ops.Num_dim)] for t in range(3) ] for mem in range(acc.Num_mem)]
        for mem in range(1,acc.Num_mem):
            for op, op_name in enumerate(['I','W','O']):
                for mapping in dataflow.tm:
                    if mem < mapping.mem[op] and acc.mappingArray[op][mem]==1:        #  <= -> <
                        trans_dim_tp[mem][op][mapping.dim] *= mapping.dimSize
                for mapping in dataflow.sm:
                    if mem <= mapping.mem[op] and acc.mappingArray[op][mem]==1:
                        trans_dim_sp[mem][op][mapping.dim] *= mapping.dimSize
        self.trans_dim_tp, self.trans_dim_sp = trans_dim_tp, trans_dim_sp
        
        tileSize = {}
        tmp_dim = [_ for _ in range(ops.Num_dim)]
        for op, op_name in enumerate(['I','W','O']):
            tileSize[acc.Num_mem, op] = 1
        for mem in range(1,acc.Num_mem):
            op = 0  # Input
            for dim in range(1,ops.Num_dim):
                tmp_dim[dim] = trans_dim_sp[mem][op][dim] * trans_dim_tp[mem][op][dim]
            if acc.mappingArray[op][mem]:
                if ops.Stride >= tmp_dim[ops.dict2Dim('R')]:
                    tmp_h = tmp_dim[ops.dict2Dim('P')] * tmp_dim[ops.dict2Dim('R')]
                else:
                    tmp_h = (tmp_dim[ops.dict2Dim('P')] - 1) * ops.Stride + tmp_dim[ops.dict2Dim('R')]
                if ops.Stride >= tmp_dim[ops.dict2Dim('S')]:
                    tmp_w = tmp_dim[ops.dict2Dim('Q')] * tmp_dim[ops.dict2Dim('S')]
                else:
                    tmp_w = (tmp_dim[ops.dict2Dim('Q')] - 1) * ops.Stride + tmp_dim[ops.dict2Dim('S')]
                tileSize[mem,op] = min(tmp_h,ops.H) * min(tmp_w,ops.W) * tmp_dim[ops.dict2Dim('C')] 
            else:
                tileSize[mem,op] = 0

            op = 1  # Weight
            for dim in range(1,ops.Num_dim):
                tmp_dim[dim] = trans_dim_sp[mem][op][dim] * trans_dim_tp[mem][op][dim]
            if acc.mappingArray[op][mem]:
                tileSize[mem,op] = tmp_dim[ops.dict2Dim('R')] * tmp_dim[ops.dict2Dim('S')] * tmp_dim[ops.dict2Dim('C')] * tmp_dim[ops.dict2Dim('K')] 
            else:
                tileSize[mem,op] = 0

            op = 2  # Output
            for dim in range(1,ops.Num_dim):
                tmp_dim[dim] = trans_dim_sp[mem][op][dim] * trans_dim_tp[mem][op][dim]
            if acc.mappingArray[op][mem]:
                tileSize[mem,op] = tmp_dim[ops.dict2Dim('P')] * tmp_dim[ops.dict2Dim('Q')] * tmp_dim[ops.dict2Dim('K')] 
            else:
                tileSize[mem,op] = 0
        for op, op_name in enumerate(['I','W','O']):
            for mem in range(1,acc.Num_mem):
                if self.dataflow.bypassMem[mem][op]:
                    tileSize[mem,op] = 0
        self.tileSize = tileSize

        self.instance_Used_count = {}
        for op, op_name in enumerate(['I','W','O']):
            for m in range(1, acc.Num_mem):
                cnt = 1
                for mapping in dataflow.sm:
                    if m > mapping.mem[op] and acc.mappingArray[op][m] == 1:
                        cnt *= mapping.dimSize
                self.instance_Used_count[m, op] = cnt
        for op, op_name in enumerate(['I','W','O']):
            self.instance_Used_count[acc.Num_mem, op] = self.instance_Used_count[acc.lastMem[1], 1]
        
        self.count_mac = 0

        self.timer = {}
        for mem in range(acc.Num_mem+1):
            for op, op_name in enumerate(['I','W','O']):
                self.timer[mem,op] = 0
        
        self.memCost = {}
        for m in range(1,acc.Num_mem+1):
            self.memCost[m] = TransCost(r=0, w=0, t=0)
        
        self.PD = ProfilingDetail()
        self.PD.memCost.append(0)

        self.PD.doubleFlag.append([])
        for mem in range(1,acc.Num_mem):
            tmp_list = []
            for op, op_name in enumerate(['I','W','O']):
                if acc.mappingArray[op][mem] == 0:
                    tmp_list.append('-') 
                else:
                    tmp_list.append(round(self.dataflow.usr_defined_double_flag[mem][op]))
            self.PD.doubleFlag.append(tmp_list)


    def ptimer(self, idx, n):
        if self.DEBUG_SIMU == False:
            return
        pstr = f"Loop {idx} iter {n}:\n"
        timer = self.timer
        for mem in range(1, self.acc.Num_mem+1):
            pstr += f"{self.acc.mem2dict(mem):<13}:"
            for op, op_name in enumerate(['I','W','O']):
                pstr += f"{op_name}({timer[mem,op]:^8})  "
            pstr += '\n'
        print(pstr)
            

    def loopExecution(self, loopidx:int):

        uppmem = self.dataflow.uppermem
        nxtmem = self.dataflow.nxtmem
        xMem   = self.dataflow.xMem

        if loopidx == len(self.dataflow.tm):
            mac_mem = self.acc.Num_mem
            # data_input_rdy = max(self.timer[mac_mem,0], self.timer[mac_mem,1])
            data_input_rdy = max(self.timer[mac_mem,op] for op in range(3))
            data_output_rdy = self.timer[mac_mem,2]
            self.ptimer(loopidx, 0)

            for op, op_name in enumerate(['I','W','O']):
                self.timer[mac_mem, op] = max(data_input_rdy + self.acc.t_MAC, data_output_rdy)

            for op, op_name in enumerate(['I','W','O']):                                # bit-serial  Macro/IOReg should be defined by user
                # if op_name == 'O':                          # Psum dont using bit-serial
                #     self.timer[mac_mem, op] += 1            # Accumulation in the merge unit 
                if self.dataflow.usr_defined_double_flag[uppmem[mac_mem,op]][op] == 0:
                    self.timer[uppmem[mac_mem,op], op] = self.timer[mac_mem, op]
            self.count_mac += 1
            
            self.ptimer(loopidx, 1)
            
        else:
            mem_cur = [self.dataflow.tm[loopidx].mem[op] for op in range(3)]
            mem_nxt = [nxtmem[mem_cur[op],op] for op in range(3)]

            nxtSize = [self.dataSize[mem_nxt[op],op] if xMem[loopidx,op]==1 else 0 for op in range(3)]
            tileSize = [self.tileSize[mem_cur[op],op] if xMem[loopidx,op]==1 else 0 for op in range(3)]

            trans = [math.ceil(tileSize[op]*self.acc.precision[mem_cur[op],op]/self.acc.bw[mem_cur[op]]) for op in range(3)]
            for op, op_name in enumerate(['I','W','O']):     
                if mem_cur[op] == self.lastMappingMem[op] and self.lastMemReg[op] is False:
                    trans[op] += 1                      # Not only 1, consider others
            op, op_name = 1,'W'     # Weight
            if mem_cur[op] == self.acc.Macro2mem:
                trans[op] = 0
            Cons = [(trans[op] if xMem[loopidx,op]==1 else 0) for op in range(3)]

            dflag = [self.dataflow.usr_defined_double_flag[mem_nxt[op]][op] for op in range(3)]
            
            self.ptimer(loopidx, 0)

            for i in range(self.dataflow.tm[loopidx].dimSize):

                for op, op_name in enumerate(['I','W','O']):
                    if dflag[op] == 1:      # next memory using double flag
                        self.timer[mem_cur[op],op] = self.timer[mem_cur[op],op] + Cons[op]
                        stall = max(0,max(self.timer[mem_nxt[op],op], self.timer[mem_cur[op],op]) - self.timer[mem_nxt[op],op])
                        self.timer[mem_nxt[op],op] = max(self.timer[mem_nxt[op],op], self.timer[mem_cur[op],op])
                    else:
                        self.timer[mem_cur[op],op] = max(self.timer[mem_cur[op],op], self.timer[mem_nxt[op],op]) + Cons[op]
                        self.timer[mem_nxt[op],op] = self.timer[mem_cur[op],op]
                        stall = Cons[op]
                    self.memCost[mem_cur[op]].t += stall

                    if mem_cur[op] == self.lastMappingMem[op] and self.lastMemReg[op] is False:
                        self.memCost[self.lastMappingMem[op]].r += self.acc.cost_r[self.lastMappingMem[op]] * tileSize[op] * self.acc.precision[self.lastMappingMem[op],op]
                        self.memCost[self.acc.lastMem[op]].w += self.acc.cost_w[self.acc.lastMem[op]] * nxtSize[op]  * self.acc.precision[self.acc.lastMem[op],op]

                        self.memCost[self.acc.lastMem[op]].r += self.acc.cost_r[self.acc.lastMem[op]] * nxtSize[op]  * self.acc.precision[self.acc.lastMem[op],op] 
                        self.memCost[mem_nxt[op]].w += self.acc.cost_w[mem_nxt[op]] * nxtSize[op]  * self.acc.precision[mem_nxt[op],op]
                    else:
                        self.memCost[mem_cur[op]].r += self.acc.cost_r[mem_cur[op]] * tileSize[op] * self.acc.precision[mem_cur[op],op] 
                        self.memCost[mem_nxt[op]].w += self.acc.cost_w[mem_nxt[op]] * nxtSize[op]  * self.acc.precision[mem_nxt[op],op]

                self.loopExecution(loopidx=loopidx+1)

                # Psum needs to be restored
                for op, op_name in enumerate(['I','W','O']):
                    if op_name == 'O':
                        if dflag[op] == 1: 
                            pass        # double-buffer makes that overlap can be placed anytime
                            # WTD
                            # self.timer
                        else:
                            self.timer[mem_nxt[op],op] = max(self.timer[mem_cur[op],op], self.timer[mem_nxt[op],op]) + Cons[op]
                            self.timer[mem_cur[op],op] = self.timer[mem_nxt[op],op]

                        if mem_cur[op] == self.lastMappingMem[op] and self.lastMemReg[op] is False:
                            self.memCost[mem_nxt[op]].r += self.acc.cost_r[mem_nxt[op]] * nxtSize[op]  * self.acc.precision[mem_nxt[op],op]
                            self.memCost[self.acc.lastMem[op]].w += self.acc.cost_w[self.acc.lastMem[op]] * nxtSize[op] * self.acc.precision[self.acc.lastMem[op],op]

                            self.memCost[self.acc.lastMem[op]].r += self.acc.cost_r[self.acc.lastMem[op]] * nxtSize[op]  * self.acc.precision[self.acc.lastMem[op],op] 
                            self.memCost[mem_cur[op]].w += self.acc.cost_w[mem_cur[op]] * tileSize[op] * self.acc.precision[mem_cur[op],op] 
                        else:
                            self.memCost[mem_nxt[op]].r += self.acc.cost_r[mem_nxt[op]] * nxtSize[op]  * self.acc.precision[mem_nxt[op],op]
                            self.memCost[mem_cur[op]].w += self.acc.cost_w[mem_cur[op]] * tileSize[op] * self.acc.precision[mem_cur[op],op]
                
                        self.memCost[mem_cur[op]].t += stall 
                self.ptimer(loopidx, i+1)

    def run(self):
        Logger.critical("Evaluation by running translation simulator") 

        self.firstMemDram = {}
        self.firstMappingMem = {}
        self.lastMemReg = {}
        self.lastMappingMem = {}
        for op, op_name in enumerate(['I','W','O']):
            self.firstMappingMem[op] = self.dataflow.tm[0].mem[op]
            self.firstMemDram[op] = True if self.dataflow.tm[0].mem[op] == self.acc.Dram2mem else False
            self.lastMappingMem[op] = self.dataflow.tm[-1].mem[op]
            self.lastMemReg[op] = True if self.lastMappingMem[op] == self.acc.lastMem[op] else False

        for op, op_name in enumerate(['I','W','O']):
            if self.firstMemDram[op] is False:
                self.timer[self.acc.Dram2mem, op]        += self.ops.size[op] * self.acc.precision[self.acc.Dram2mem,op] / self.acc.bw[self.acc.Dram2mem]
                self.timer[self.firstMappingMem[op], op] += self.ops.size[op] * self.acc.precision[self.acc.Dram2mem,op] / self.acc.bw[self.acc.Dram2mem]

        self.loopExecution(0)

        for op, op_name in enumerate(['I','W','O']):
            if op_name == 'O' and self.firstMemDram[op] is False:
                self.timer[self.acc.Dram2mem, op]        += self.ops.size[op] * self.acc.precision[self.acc.Dram2mem,op] / self.acc.bw[self.acc.Dram2mem]
                self.timer[self.firstMappingMem[op], op] += self.ops.size[op] * self.acc.precision[self.acc.Dram2mem,op] / self.acc.bw[self.acc.Dram2mem]

        res_Latency = max(self.timer[mem, op] for op in range(3) for mem in range(self.acc.Num_mem))

        # ['PLACEHOLD'0, 'Dram'1, 'Global_buffer'2, 'Output_buffer'3, 'Input_buffer'4, 'OReg'5, 'IReg'6, 'Macro'7] acc.Num_mem = 8
        
        for op, op_name in enumerate(['I','W','O']):
            if self.firstMemDram[op] is False:
                self.memCost[self.acc.Dram2mem].r += self.dataSize[self.firstMappingMem[op],op] * \
                                                    self.acc.cost_r[self.acc.Dram2mem] * self.acc.precision[self.acc.Dram2mem,op]
                self.memCost[self.firstMappingMem[op]].w += self.dataSize[self.firstMappingMem[op],op] * \
                                                       self.acc.cost_w[self.firstMappingMem[op]] * self.acc.precision[self.firstMappingMem[op],op]
        
        for m in range(1, self.acc.Num_mem):
            for op, op_name in enumerate(['I','W','O']):
                if self.acc.mappingArray[op][m] == 1 and self.instance_Used_count[m,op] > 1:
                    self.memCost[m].r *= self.instance_Used_count[m,op]
                    self.memCost[m].w *= self.instance_Used_count[m,op]
                    continue

        self.memCost[self.acc.Macro2mem].r = 0

        cost_r, cost_w = 0, 0
        for m in range(1,self.acc.Num_mem):
            self.PD.memCost.append(0)
            cost_r += self.memCost[m].r
            cost_w += self.memCost[m].w
            self.PD.memCost[m] += self.memCost[m].r + self.memCost[m].w

        cost_m = self.acc.cost_ActMacro * self.count_mac * self.instance_Used_count[4,0]
                                                           # the number of multi-core is equal with the number of input buffer instance

        cost_s = res_Latency * self.acc.leakage_per_cycle

        res_Energy = cost_r+cost_w+cost_m+cost_s
        
        Logger.info(f"* * SimuLax-Running  * *  Latency:{res_Latency}, MAC-times:{self.count_mac}, MAC-Latency:{self.count_mac*self.acc.t_MAC}, MAC-Energy:{cost_m:.3e} nJ")
        Logger.info(f"* * SimuLax-Running  * *  EnergyCost:{round(res_Energy,3)}, Dynamic({cost_r+cost_w+cost_m}), Leakage:{round(cost_s,3)}")
        # Logger.info(f"* * SimuLax-Running  * *  readOUT:{round(cost_r,3)}, writeIN:{round(cost_w,3)}, MAC:{round(cost_m,3)}")
        Logger.info(f"* * SimuLax-Running  * *  Memory Cost:") 
        for m in range(1,self.acc.Num_mem):
            Logger.info('-'*4 + f" {self.acc.mem2dict(m)}: {(self.memCost[m].r + self.memCost[m].w):.3e} nJ" +
                        f" {self.memCost[m].r:.3e}(Read) + {self.memCost[m].w:.3e}(Write), "
                        f" ({round((self.memCost[m].r + self.memCost[m].w)/res_Energy*100,2)}%)")
        Logger.info(f"* * SimuLax-Running  * *  Energy BreakDown: " + 
                    f"MemoryAccess(On/Off):{round((cost_r+cost_w-(self.memCost[1].r+self.memCost[1].w))/res_Energy*100,2)}%/{round((self.memCost[1].r+self.memCost[1].w)/res_Energy*100,2)}%, " +
                    f"MAC:{round(cost_m/res_Energy*100,2)}%, " + 
                    f"Leakage:{round(cost_s/res_Energy*100,2)}%" ) 
        Logger.info(f"* * SimuLax-Running  * *  EDP:{((res_Energy)*res_Latency):.3e}")
        Logger.info('* '*50+'\n')

        self.PD.dynamic_power = cost_r+cost_w+cost_m
        self.PD.dynamic_power_onChip = self.PD.dynamic_power - self.PD.memCost[1]
        self.PD.leakage_power = cost_s
        self.PD.macEnergy = cost_m

        self.PD.latency = res_Latency
        self.PD.macLatency = self.count_mac * self.acc.t_MAC

        return res_Latency, res_Energy

        
    def LModeling(self):
        acc = self.acc
        ops = self.ops
        tm = self.dataflow.tm
        uppmem = self.dataflow.uppermem
        nxtmem = self.dataflow.nxtmem
        xMem   = self.dataflow.xMem
        Lcp = {}
        Lcons = {}
        cp_term = {}
        debugStrTrans = {}
        for op, op_name in enumerate(['I','W','O']):
            Lcons[len(tm),op] = acc.t_MAC
        cp_term[len(tm)] = max(Lcons[len(tm),op] for op, op_name in enumerate(['I','W','O']))
        for loopidx in range(len(tm)-1,-1,-1):
            mem_cur = [self.dataflow.tm[loopidx].mem[op] for op in range(3)]
            mem_nxt = [nxtmem[mem_cur[op],op] for op in range(3)]
            tileSize = [self.tileSize[mem_cur[op],op] for op in range(3)]
            trans = [math.ceil(tileSize[op]*self.acc.precision[mem_cur[op],op]/self.acc.bw[mem_cur[op]]) for op in range(3)]

            if loopidx == len(tm)-1:
                for op, op_name in enumerate(['I','W','O']):
                    if mem_cur[op] != acc.lastMem[op]:
                        trans[op] += 1

            Trans = [(trans[op] if xMem[loopidx,op]==1 else 0) for op in range(3)]
            Trans[2] *= 2       # for output operands
            dflag = [self.dataflow.usr_defined_double_flag[mem_nxt[op]][op] for op in range(3)]

            # if loopidx == len(tm)-1:
            #     Lcp[loopidx] = Lcp[loopidx+1]*1
            # else:
            #     Lcp[loopidx] = Lcp[loopidx+1]*tm[loopidx+1].dimSize
            Lcp[loopidx] = cp_term[loopidx+1]

            for op, op_name in enumerate(['I','W','O']):
                Lcp[loopidx] = max(Lcp[loopidx], (Trans[op]+Lcons[loopidx+1,op])*(1-dflag[op])+max(Trans[op],Lcons[loopidx+1,op])*dflag[op])
            
            for op, op_name in enumerate(['I','W','O']):
                if xMem[loopidx,op]==0:
                    Lcons[loopidx,op] = Lcp[loopidx]*(tm[loopidx].dimSize-1) + Lcons[loopidx+1,op]
                else:
                    if dflag[op]==0:
                        if op < 2:  # I,W
                            Lcons[loopidx,op] = Lcp[loopidx]*(tm[loopidx].dimSize-2) + 2*Trans[op] + Lcons[loopidx+1,op] 
                        else:
                            Lcons[loopidx,op] = Lcp[loopidx]*(tm[loopidx].dimSize-1) + Trans[op] + Lcons[loopidx+1,op]
                    else:
                        if op < 2:  # I,W
                            Lcons[loopidx,op] = max(Lcp[loopidx]*max(0,tm[loopidx].dimSize-3) + 2*Trans[op] + max(Trans[op],Lcons[loopidx+1,op]), Trans[op]*tm[loopidx].dimSize)
                        else:
                            Lcons[loopidx,op] = Lcp[loopidx]*(tm[loopidx].dimSize-2) + Trans[op] + max(math.ceil(Trans[op]/2),Lcons[loopidx+1,op]) + max(math.ceil(Trans[op]/2),Lcp[loopidx]) 
            
            cp_term[loopidx] = Lcp[loopidx]*tm[loopidx].dimSize
                
            pstr = ""
            for op, op_name in enumerate(['I','W','O']):
                pstr += f"{round(dflag[op])}|{Trans[op]:<10}   "
            pstr += f"Lcp:{round(Lcp[loopidx])}"
            debugStrTrans[loopidx] = pstr
            
        res_Latency = max(cp_term[0], max(Lcons[0,op] for op in range(3)))
        Logger.info(f"* * SimuLax-Latency-Modeling * *  Latency:{res_Latency}")

        print("\nDebug in Latency Modeling")
        print("doubleFlag | tileSize(bit)")
        # self.dataflow.usr_defined_double_flag[mem_nxt[op]][op]
        dflag = self.dataflow.usr_defined_double_flag
        for m in range(1,acc.Num_mem):
            print(f"{acc.mem2dict(m):<15}: ",end="")
            for op, op_name in enumerate(['I','W','O']):
                print(f"{round(dflag[m][op])}|{self.tileSize[m,op]*acc.precision[m,op]:<10} ", end="")
            print("")

        print("dFlag | TransTime")
        for loopidx in range(len(tm)):
            print(f"loop {loopidx}({tm[loopidx].dimSize}): ",end="")
            print(debugStrTrans[loopidx])
        # print(f"doubleBuffer flag: {self.dataflow.usr_defined_double_flag}")
        
        print("Latency")
        for loopidx in range(len(tm)):
            # print(f"loop {loopidx}({tm[loopidx].dimSize}): Lcp={Lcp[loopidx]}, ",end="")
            print(f"loop {loopidx}({tm[loopidx].dimSize}):",end="   ")
            for op, op_name in enumerate(['I','W','O']):
                print(f"{Lcons[loopidx,op]:<12}  ", end="")
            print(f"critical_term={cp_term[loopidx]}")

        return res_Latency

    def debugLog(self):
        acc = self.acc
        ops = self.ops
        pstr = "\n"
        
        str_tpsp = "dim_tp & dim_sp \n"
        for mem in range(1,acc.Num_mem):
            # str_tpsp += '- ' * 10 + f"{acc.mem2dict(mem)}: dim_tp & dim_sp " + '- ' * 10 + '\n'
            str_tpsp += '- - - - - tp - - - - - ' + f"{acc.mem2dict(mem):^15}" + '- - - - - sp - - - - - ' + '\n'
            for op, op_name in enumerate(['I','W','O']):
                str_tpsp += f"{op_name}:"
                for dim in range(1,ops.Num_dim):
                    str_tpsp += f"{ops.dim2Dict[dim]}:{self.dim_tp[mem][op][dim]} "
                str_tpsp += '    |    '
                for dim in range(1,ops.Num_dim):
                    str_tpsp += f"{ops.dim2Dict[dim]}:{self.dim_sp[mem][op][dim]} "
                str_tpsp += '\n'
        # pstr += str_tpsp

        
        str_mem = "Data element size:\n"
        for mem in range(1,acc.Num_mem):
            str_mem += f"{acc.mem2dict(mem):<15}:"
            for t in range(3):
                str_mem += f"{self.dataSize[mem,t]:>8}, {int(self.multicast_size[mem,t]):>3}|{int(self.unicast_size[mem,t]):<3}"
                # str_mem += f"{self.dataSize[mem,t]:>8}"
            str_mem += '\n'
        pstr += str_mem
        
        str_relat_mem = "\n"
        for mem in range(1, acc.Num_mem):
            for op, op_name in enumerate(['I','W','O']):    
                if self.dataflow.bypassMem[mem][op] == 0:
                    str_relat_mem += f"{acc.mem2dict(mem)}({op_name}): "
                    str_relat_mem += f"next_mem is {acc.mem2dict(self.dataflow.nxtmem[mem,op])}"
                    str_relat_mem += '\n'
        # pstr += str_relat_mem


        return pstr



        