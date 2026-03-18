from dataclasses import dataclass, field
import math

from Simulator.Simulax import ProfilingDetail, TransCost, tranSimulator
from utils.GlobalUT import Logger


@dataclass
class AccessEvent:
    phase: str
    op: str
    mem: str
    direction: str
    words: float
    energy: float
    loopidx: int = -1
    iter_idx: int = -1
    peer_mem: str = ""
    tile_size: float = 0
    nxt_size: float = 0
    transfer_cycles: float = 0
    stall: float = 0
    note: str = ""


@dataclass
class MemAccessDebug:
    read_words: float = 0
    write_words: float = 0
    read_energy: float = 0
    write_energy: float = 0
    read_events: int = 0
    write_events: int = 0
    transfer_stall: float = 0


@dataclass
class LoopDebugSnapshot:
    loopidx: int
    iter_idx: int
    op: str
    mem_cur: str
    mem_nxt: str
    xmem: int
    dflag: int
    tile_size: float
    nxt_size: float
    transfer_cycles: float


class tranSimulatorDebuger(tranSimulator):
    """
    A debugging-oriented simulator that mirrors Simulax behavior while recording
    detailed access events and per-memory read/write counts.
    """

    def __init__(self, acc, ops, dataflow, DEBUG_SIMU=False):
        super().__init__(acc, ops, dataflow, DEBUG_SIMU=DEBUG_SIMU)
        self.access_events: list[AccessEvent] = []
        self.loop_snapshots: list[LoopDebugSnapshot] = []
        self.mem_debug = {}
        for m in range(1, self.acc.Num_mem + 1):
            for op, op_name in enumerate(["I", "W", "O"]):
                self.mem_debug[m, op] = MemAccessDebug()

    def _record_access(self, phase, op, mem, direction, words, energy,
                       loopidx=-1, iter_idx=-1, peer_mem="", tile_size=0, nxt_size=0,
                       transfer_cycles=0, stall=0, note=""):
        if words == 0 and energy == 0 and stall == 0:
            return

        op_name = ["I", "W", "O"][op]
        mem_name = self.acc.mem2dict(mem)
        stats = self.mem_debug[mem, op]
        if direction == "read":
            stats.read_words += words
            stats.read_energy += energy
            stats.read_events += 1
        elif direction == "write":
            stats.write_words += words
            stats.write_energy += energy
            stats.write_events += 1
        stats.transfer_stall += stall

        self.access_events.append(
            AccessEvent(
                phase=phase,
                op=op_name,
                mem=mem_name,
                direction=direction,
                words=words,
                energy=energy,
                loopidx=loopidx,
                iter_idx=iter_idx,
                peer_mem=self.acc.mem2dict(peer_mem) if peer_mem else "",
                tile_size=tile_size,
                nxt_size=nxt_size,
                transfer_cycles=transfer_cycles,
                stall=stall,
                note=note,
            )
        )

    def _record_loop_snapshot(self, loopidx, iter_idx, op, mem_cur, mem_nxt, xmem, dflag, tile_size, nxt_size, transfer_cycles):
        self.loop_snapshots.append(
            LoopDebugSnapshot(
                loopidx=loopidx,
                iter_idx=iter_idx,
                op=["I", "W", "O"][op],
                mem_cur=self.acc.mem2dict(mem_cur),
                mem_nxt=self.acc.mem2dict(mem_nxt),
                xmem=int(xmem),
                dflag=int(dflag),
                tile_size=tile_size,
                nxt_size=nxt_size,
                transfer_cycles=transfer_cycles,
            )
        )

    def loopExecution(self, loopidx: int):
        uppmem = self.dataflow.uppermem
        nxtmem = self.dataflow.nxtmem
        xMem = self.dataflow.xMem

        if loopidx == len(self.dataflow.tm):
            mac_mem = self.acc.Num_mem
            data_input_rdy = max(self.timer[mac_mem, op] for op in range(3))
            data_output_rdy = self.timer[mac_mem, 2]
            self.ptimer(loopidx, 0)

            for op in range(3):
                self.timer[mac_mem, op] = max(data_input_rdy + self.acc.t_MAC, data_output_rdy)

            for op in range(3):
                if self.dataflow.usr_defined_double_flag[uppmem[mac_mem, op]][op] == 0:
                    self.timer[uppmem[mac_mem, op], op] = self.timer[mac_mem, op]
            self.count_mac += 1
            self.ptimer(loopidx, 1)
            return

        mem_cur = [self.dataflow.tm[loopidx].mem[op] for op in range(3)]
        mem_nxt = [nxtmem[mem_cur[op], op] for op in range(3)]

        nxtSize = [self.dataSize[mem_nxt[op], op] if xMem[loopidx, op] == 1 else 0 for op in range(3)]
        tileSize = [self.tileSize[mem_cur[op], op] if xMem[loopidx, op] == 1 else 0 for op in range(3)]

        trans = [math.ceil(tileSize[op] * self.acc.precision[mem_cur[op], op] / self.acc.bw[mem_cur[op]]) for op in range(3)]
        for op in range(3):
            if mem_cur[op] == self.lastMappingMem[op] and self.lastMemReg[op] is False:
                trans[op] += 1
        if mem_cur[1] == self.acc.Macro2mem:
            trans[1] = 0
        Cons = [(trans[op] if xMem[loopidx, op] == 1 else 0) for op in range(3)]
        dflag = [self.dataflow.usr_defined_double_flag[mem_nxt[op]][op] for op in range(3)]

        self.ptimer(loopidx, 0)

        for i in range(self.dataflow.tm[loopidx].dimSize):
            for op in range(3):
                self._record_loop_snapshot(
                    loopidx, i, op, mem_cur[op], mem_nxt[op], xMem[loopidx, op], dflag[op],
                    tileSize[op], nxtSize[op], Cons[op]
                )

                if dflag[op] == 1:
                    self.timer[mem_cur[op], op] = self.timer[mem_cur[op], op] + Cons[op]
                    stall = max(0, max(self.timer[mem_nxt[op], op], self.timer[mem_cur[op], op]) - self.timer[mem_nxt[op], op])
                    self.timer[mem_nxt[op], op] = max(self.timer[mem_nxt[op], op], self.timer[mem_cur[op], op])
                else:
                    self.timer[mem_cur[op], op] = max(self.timer[mem_cur[op], op], self.timer[mem_nxt[op], op]) + Cons[op]
                    self.timer[mem_nxt[op], op] = self.timer[mem_cur[op], op]
                    stall = Cons[op]
                self.memCost[mem_cur[op]].t += stall

                if mem_cur[op] == self.lastMappingMem[op] and self.lastMemReg[op] is False:
                    energy = self.acc.cost_r[self.lastMappingMem[op]] * tileSize[op] * self.acc.precision[self.lastMappingMem[op], op]
                    self.memCost[self.lastMappingMem[op]].r += energy
                    self._record_access("forward_last_stage", op, self.lastMappingMem[op], "read", tileSize[op], energy,
                                        loopidx, i, mem_nxt[op], tileSize[op], nxtSize[op], Cons[op], stall)

                    energy = self.acc.cost_w[self.acc.lastMem[op]] * nxtSize[op] * self.acc.precision[self.acc.lastMem[op], op]
                    self.memCost[self.acc.lastMem[op]].w += energy
                    self._record_access("forward_last_stage", op, self.acc.lastMem[op], "write", nxtSize[op], energy,
                                        loopidx, i, mem_cur[op], tileSize[op], nxtSize[op], Cons[op], stall)

                    energy = self.acc.cost_r[self.acc.lastMem[op]] * nxtSize[op] * self.acc.precision[self.acc.lastMem[op], op]
                    self.memCost[self.acc.lastMem[op]].r += energy
                    self._record_access("forward_last_stage", op, self.acc.lastMem[op], "read", nxtSize[op], energy,
                                        loopidx, i, mem_cur[op], tileSize[op], nxtSize[op], Cons[op], stall, note="last_mem_stage_restore_read")

                    energy = self.acc.cost_w[mem_nxt[op]] * nxtSize[op] * self.acc.precision[mem_nxt[op], op]
                    self.memCost[mem_nxt[op]].w += energy
                    self._record_access("forward_last_stage", op, mem_nxt[op], "write", nxtSize[op], energy,
                                        loopidx, i, mem_cur[op], tileSize[op], nxtSize[op], Cons[op], stall, note="last_mem_stage_forward_write")
                else:
                    energy = self.acc.cost_r[mem_cur[op]] * tileSize[op] * self.acc.precision[mem_cur[op], op]
                    self.memCost[mem_cur[op]].r += energy
                    self._record_access("forward", op, mem_cur[op], "read", tileSize[op], energy,
                                        loopidx, i, mem_nxt[op], tileSize[op], nxtSize[op], Cons[op], stall)

                    energy = self.acc.cost_w[mem_nxt[op]] * nxtSize[op] * self.acc.precision[mem_nxt[op], op]
                    self.memCost[mem_nxt[op]].w += energy
                    self._record_access("forward", op, mem_nxt[op], "write", nxtSize[op], energy,
                                        loopidx, i, mem_cur[op], tileSize[op], nxtSize[op], Cons[op], stall)

            self.loopExecution(loopidx=loopidx + 1)

            for op in range(3):
                if op != 2:
                    continue
                if dflag[op] == 1:
                    self.timer[mem_cur[op], op] += Cons[op]
                    self.timer[mem_nxt[op], op] = max(self.timer[mem_nxt[op], op], self.timer[mem_cur[op], op])
                    stall = 0
                else:
                    self.timer[mem_nxt[op], op] = max(self.timer[mem_cur[op], op], self.timer[mem_nxt[op], op]) + Cons[op]
                    self.timer[mem_cur[op], op] = self.timer[mem_nxt[op], op]
                    stall = Cons[op]

                if mem_cur[op] == self.lastMappingMem[op] and self.lastMemReg[op] is False:
                    energy = self.acc.cost_r[mem_nxt[op]] * nxtSize[op] * self.acc.precision[mem_nxt[op], op]
                    self.memCost[mem_nxt[op]].r += energy
                    self._record_access("restore_last_stage", op, mem_nxt[op], "read", nxtSize[op], energy,
                                        loopidx, i, mem_cur[op], tileSize[op], nxtSize[op], Cons[op], stall)

                    energy = self.acc.cost_w[self.acc.lastMem[op]] * nxtSize[op] * self.acc.precision[self.acc.lastMem[op], op]
                    self.memCost[self.acc.lastMem[op]].w += energy
                    self._record_access("restore_last_stage", op, self.acc.lastMem[op], "write", nxtSize[op], energy,
                                        loopidx, i, mem_cur[op], tileSize[op], nxtSize[op], Cons[op], stall)

                    energy = self.acc.cost_r[self.acc.lastMem[op]] * nxtSize[op] * self.acc.precision[self.acc.lastMem[op], op]
                    self.memCost[self.acc.lastMem[op]].r += energy
                    self._record_access("restore_last_stage", op, self.acc.lastMem[op], "read", nxtSize[op], energy,
                                        loopidx, i, mem_cur[op], tileSize[op], nxtSize[op], Cons[op], stall, note="last_mem_stage_restore_read")

                    energy = self.acc.cost_w[mem_cur[op]] * tileSize[op] * self.acc.precision[mem_cur[op], op]
                    self.memCost[mem_cur[op]].w += energy
                    self._record_access("restore_last_stage", op, mem_cur[op], "write", tileSize[op], energy,
                                        loopidx, i, mem_nxt[op], tileSize[op], nxtSize[op], Cons[op], stall)
                else:
                    energy = self.acc.cost_r[mem_nxt[op]] * nxtSize[op] * self.acc.precision[mem_nxt[op], op]
                    self.memCost[mem_nxt[op]].r += energy
                    self._record_access("restore", op, mem_nxt[op], "read", nxtSize[op], energy,
                                        loopidx, i, mem_cur[op], tileSize[op], nxtSize[op], Cons[op], stall)

                    energy = self.acc.cost_w[mem_cur[op]] * tileSize[op] * self.acc.precision[mem_cur[op], op]
                    self.memCost[mem_cur[op]].w += energy
                    self._record_access("restore", op, mem_cur[op], "write", tileSize[op], energy,
                                        loopidx, i, mem_nxt[op], tileSize[op], nxtSize[op], Cons[op], stall)

                self.memCost[mem_cur[op]].t += stall
            self.ptimer(loopidx, i + 1)

    def run(self):
        Logger.critical("Evaluation by running translation simulator debugger")

        self.firstMemDram = {}
        self.firstMappingMem = {}
        self.lastMemReg = {}
        self.lastMappingMem = {}
        for op in range(3):
            self.firstMappingMem[op] = self.dataflow.tm[0].mem[op]
            self.firstMemDram[op] = self.dataflow.tm[0].mem[op] == self.acc.Dram2mem
            self.lastMappingMem[op] = self.dataflow.tm[-1].mem[op]
            self.lastMemReg[op] = self.lastMappingMem[op] == self.acc.lastMem[op]

        for op in range(3):
            if self.firstMemDram[op] is False:
                words = self.ops.size[op]
                cycles = words * self.acc.precision[self.acc.Dram2mem, op] / self.acc.bw[self.acc.Dram2mem]
                self.timer[self.acc.Dram2mem, op] += cycles
                self.timer[self.firstMappingMem[op], op] += cycles

        self.loopExecution(0)

        for op in range(3):
            if op == 2 and self.firstMemDram[op] is False:
                words = self.ops.size[op]
                cycles = words * self.acc.precision[self.acc.Dram2mem, op] / self.acc.bw[self.acc.Dram2mem]
                self.timer[self.acc.Dram2mem, op] += cycles
                self.timer[self.firstMappingMem[op], op] += cycles

        res_Latency = max(self.timer[mem, op] for op in range(3) for mem in range(self.acc.Num_mem))

        for op in range(3):
            if self.firstMemDram[op] is False:
                words = self.dataSize[self.firstMappingMem[op], op]

                energy = words * self.acc.cost_r[self.acc.Dram2mem] * self.acc.precision[self.acc.Dram2mem, op]
                self.memCost[self.acc.Dram2mem].r += energy
                self._record_access("bootstrap", op, self.acc.Dram2mem, "read", words, energy, peer_mem=self.firstMappingMem[op])

                energy = words * self.acc.cost_w[self.firstMappingMem[op]] * self.acc.precision[self.firstMappingMem[op], op]
                self.memCost[self.firstMappingMem[op]].w += energy
                self._record_access("bootstrap", op, self.firstMappingMem[op], "write", words, energy, peer_mem=self.acc.Dram2mem)

        for m in range(1, self.acc.Num_mem):
            for op in range(3):
                if self.acc.mappingArray[op][m] == 1 and self.instance_Used_count[m, op] > 1:
                    scale = self.instance_Used_count[m, op]
                    self.memCost[m].r *= scale
                    self.memCost[m].w *= scale
                    self.mem_debug[m, op].read_words *= scale
                    self.mem_debug[m, op].write_words *= scale
                    self.mem_debug[m, op].read_energy *= scale
                    self.mem_debug[m, op].write_energy *= scale

        self.memCost[self.acc.Macro2mem].r = 0
        self.mem_debug[self.acc.Macro2mem, 1].read_energy = 0
        self.mem_debug[self.acc.Macro2mem, 1].read_words = 0

        cost_r, cost_w = 0, 0
        for m in range(1, self.acc.Num_mem):
            cost_r += self.memCost[m].r
            cost_w += self.memCost[m].w

        cost_m = self.acc.cost_ActMacro * self.count_mac * self.instance_Used_count[4, 0]
        cost_s = res_Latency * self.acc.leakage_per_cycle
        res_Energy = cost_r + cost_w + cost_m + cost_s

        self.PD.dynamic_power = cost_r + cost_w + cost_m
        self.PD.dynamic_power_onChip = self.PD.dynamic_power - self.memCost[1].r - self.memCost[1].w
        self.PD.leakage_power = cost_s
        self.PD.macEnergy = cost_m
        self.PD.latency = res_Latency
        self.PD.macLatency = self.count_mac * self.acc.t_MAC

        return res_Latency, res_Energy

    def get_mem_access_summary(self):
        summary = {}
        for m in range(1, self.acc.Num_mem + 1):
            mem_name = self.acc.mem2dict(m)
            summary[mem_name] = {}
            for op, op_name in enumerate(["I", "W", "O"]):
                stats = self.mem_debug[m, op]
                summary[mem_name][op_name] = {
                    "read_words": stats.read_words,
                    "write_words": stats.write_words,
                    "read_energy": stats.read_energy,
                    "write_energy": stats.write_energy,
                    "read_events": stats.read_events,
                    "write_events": stats.write_events,
                    "transfer_stall": stats.transfer_stall,
                }
        return summary

    def print_debug_summary(self):
        Logger.info("* * SimuDebuger * * Memory access summary")
        for mem_name, ops in self.get_mem_access_summary().items():
            Logger.info(f"[{mem_name}]")
            for op_name in ["I", "W", "O"]:
                stats = ops[op_name]
                Logger.info(
                    f"  {op_name}: read_words={stats['read_words']}, write_words={stats['write_words']}, "
                    f"read_energy={stats['read_energy']:.6f}, write_energy={stats['write_energy']:.6f}, "
                    f"read_events={stats['read_events']}, write_events={stats['write_events']}, "
                    f"stall={stats['transfer_stall']}"
                )


DebugTranSimulator = tranSimulatorDebuger
