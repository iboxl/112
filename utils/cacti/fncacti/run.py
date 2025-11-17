import subprocess
import re


def run_cacti():
	
    command = "./cacti -infile configs/config.cfg > test.log"
    with open("test.log", "w") as log_file:
        subprocess.call(command.split(), stdout=log_file)

        with open("test.log", "r") as log_file:
            log_content = log_file.read()
        
    read_energy_pattern = r"Total dynamic read energy per access \(nJ\):.*?(\d+\.\d+)(?:;|\s)*$"
    write_energy_pattern = r"Total dynamic write energy per access \(nJ\):.*?(\d+\.\d+)(?:;|\s)*$"
    leakage_power_pattern = r"Total leakage power of a bank \(mW\):.*?(\d+\.\d+)(?:;|\s)*$"

    read_energy_match = re.findall(read_energy_pattern, log_content, re.MULTILINE)
    write_energy_match = re.findall(write_energy_pattern, log_content, re.MULTILINE)
    leakage_power_match = re.findall(leakage_power_pattern, log_content, re.MULTILINE)

    read_energy = float(read_energy_match[-1]) if read_energy_match else None
    write_energy = float(write_energy_match[-1]) if write_energy_match else None
    leakage_power = float(leakage_power_match[-1]) if leakage_power_match else None

    #er = float(read_energy.group(1))
    er = float(read_energy)
    er_b = round(er*1e3, 3)
    #ew = float(write_energy.group(1))
    ew = float(write_energy)
    ew_b = round(ew*1e3, 3)
    #pl = float(leakage_power.group(1))
    pl = float(leakage_power)
    el = round(pl * 2 * 4, 3)	# 500Mhz
    if read_energy:
        print("Total dynamic read energy per access (pJ): {0}".format(er_b))
    if write_energy:
        print("Total dynamic write energy per access (pJ): {0}".format(ew_b))
    if leakage_power:
        print("leakage per cycle (pJ): {0}".format(el))
    return er_b, ew_b, el

if __name__ == "__main__":
    run_cacti()