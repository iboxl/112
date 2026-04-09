import pickle

from baseline.types import BaselineLayer
from utils.GlobalUT import Logger
from utils.ZigzagUtils import (
    zigzag_cache_prefix,
    get_hardware_performance_zigzag,
)
from Evaluation.Zigzag_imc.CompatibleZigzag import (
    compare_ops_cme,
    baseline_layer_from_zigzag_cme,
)


class ZigzagBaselineAdapter:
    def __init__(self, model: str, architecture: str, opt_flag: str):
        self.model = model
        self.architecture = architecture
        self.opt_flag = opt_flag
        self._cmes = self._load_or_generate_cmes()

    def _load_or_generate_cmes(self):
        compare_file_prefix = zigzag_cache_prefix(self.opt_flag, self.model, self.architecture)
        compare_pickle = compare_file_prefix.with_suffix(".pickle")
        compare_json = compare_file_prefix.with_suffix(".json")

        if compare_pickle.is_file() is False:
            Logger.info("Running Zigzag to generate baseline mappings")
            get_hardware_performance_zigzag(
                workload=f"model/{self.model}.onnx",
                accelerator=f"Architecture.{self.architecture}",
                mapping="Config.zigzag_mapping",
                opt=self.opt_flag,
                dump_filename_pattern=str(compare_json),
                pickle_filename=str(compare_pickle),
            )

        with open(compare_pickle, "rb") as fp:
            return pickle.load(fp)

    def find_layer(self, loop_dim: dict[str, int]) -> BaselineLayer:
        cme = next((c for c in self._cmes if compare_ops_cme(loopDim=loop_dim, cme=c)), None)
        if cme is None:
            raise ValueError(f"No ZigZag baseline layer found for loop_dim={loop_dim}")
        return baseline_layer_from_zigzag_cme(cme)
