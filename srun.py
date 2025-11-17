# this file is prepared for project 026
# Created by iboxl


import configparser
import subprocess
import itertools
import os
from pprint import pformat

def skip_by(experiment_params):
    if (experiment_params['Macro']['compartment'] > 50 and experiment_params['Macro']['cell'] > 1) \
        or (experiment_params['Macro']['compartment'] < 50 and experiment_params['Macro']['cell'] == 1) :
        # or (experiment_params['Accelerator']['num_core'] <= 8 and (experiment_params['Accelerator']['bandwidth_global_to_iBuffer'] > 32 or 
        #                                                            experiment_params['Accelerator']['bandwidth_oBuffer_to_global'] > 32) ) \
        # or (experiment_params['Macro']['compartment']==16 and experiment_params['Macro']['cell'] == 16 and experiment_params['Accelerator']['num_core'] == 8   \
        #     and experiment_params['Accelerator']['bandwidth_global_to_iBuffer'] == 16 and experiment_params['Accelerator']['bandwidth_oBuffer_to_global'] == 16 \
        #         and experiment_params['Core']['size_output_buffer'] == 2) \
        # or False:
        return True
    if (experiment_params['Accelerator']['bandwidth_oBuffer_to_global'] != experiment_params['Accelerator']['bandwidth_global_to_iBuffer'] ):
        return True
    # if (experiment_params['Workload']['weight_bit_width'] == 8 and experiment_params['Workload']['input_bit_width'] == 8 ):
    #     return True
    else:
        return False
    return False

# 初始化config解析器
config = configparser.ConfigParser()
log_file = os.getcwd() + '/srun_new.log'
config_file = os.getcwd() + '/Config/cim_template.cfg'
experiment_dir = os.getcwd() + '/Config/'
experiment_file = 'cim_test_1.cfg'
with open(log_file, 'w'):
    pass

# 定义参数取值范围
"""
param_dict = {
    'Macro': {
        'compartment': [16, 32, 64, 128],
        # 'cell': [1, 16, 32, 64]
        'column': [128, 256],
        'cell': [1, 16, 32, 64]
    },
    'Accelerator': {
        'num_core': [4, 8, 16, 32],
        'bandwidth_global_to_iBuffer': [16,32,40,80],
        # 'bandwidth_global_to_macro': [16,32,40,80],
        'bandwidth_oBuffer_to_global': [16,32,40,80],
    },
    'Core': {
        # 'size_input_buffer': [2, 4, 8, 16, 32],
        # 'size_input_buffer': [2, 8, 16],
        # 'size_output_buffer': [2, 4, 8, 16, 32],
        'size_output_buffer': [2, 8, 16],
    },
    # 'Workload':{
        # 'weight_bit_width': [4,8],
        # 'weight_bit_width': [8],
        # 'input_bit_width': [1,2,4,8]
        # 'input_bit_width': [8]
    # }
}
"""
param_dict = {
    # 'Macro': {
    #     'compartment': [16, 32, 128],
    #     'cell': [1, 16, 32, 64]
    # },
    'Accelerator': {
        # 'num_core': [4, 8, 16, 32],
        'bandwidth_global_to_iBuffer': [32, 64 ,128, 256, 512],
        'bandwidth_oBuffer_to_global': [32, 64 ,128, 256, 512],
    },
    'Core': {
        'size_output_buffer': [2, 4, 8, 16],
    },
}


param_combinations = [list(itertools.product(*params.values())) for params in param_dict.values()]
param_names = list(param_dict.keys())
param_values = list(itertools.product(*param_dict.values()))

# 计算所有可能的参数组合的数量
total_combinations = 1
for comb in param_combinations:
    total_combinations *= len(comb)
iter = 0
for values in itertools.product(*param_combinations):
    config.read(config_file)
    experiment_params = {}
    for section, section_values in zip(param_dict.keys(), values):
        section_params = {}
        for name, value in zip(param_dict[section].keys(), section_values):
            config.set(section, name, str(value))
            section_params[name] = value
        experiment_params[section] = section_params
    if skip_by(experiment_params=experiment_params):
        continue
    else:
        iter+=1

# 在日志文件的开始处打印总的参数组合数量
with open(log_file, 'w') as f:
    f.write("\n" + "="*50 + "\n")
    f.write(f"Total number of parameter combinations: {total_combinations} ({iter})\n")
    f.write("="*50 + "\n\n")

# 遍历所有参数组合
iter = 0
for values in itertools.product(*param_combinations):
    config.read(config_file)

    # 更新配置文件的参数
    experiment_params = {}
    for section, section_values in zip(param_dict.keys(), values):
        section_params = {}
        for name, value in zip(param_dict[section].keys(), section_values):
            config.set(section, name, str(value))
            section_params[name] = value
        experiment_params[section] = section_params
    
    if skip_by(experiment_params=experiment_params):
        continue

    # 保存更新后的配置文件
    with open(experiment_dir+experiment_file, 'w') as configfile:
        config.write(configfile)
        
    iter += 1
    with open(log_file, 'a') as f:
        f.write(f"Experiment {iter} with parameters:\n")
        f.write(pformat(experiment_params))
        f.write("\n" + "="*50 + "\n")

    cmd_args = [
            'python', 'run.py', '--logger', '--srun', '--noLogFile',
            # 'python', 'run.py', '--noLogFile',
            # '--debug',
            '--SIMU',
            # '-m', 'mob',
            '-c', experiment_file,
        ]

    with open(log_file, 'a') as f:
    #         subprocess.run(cmd_args+['-opt', '0'], stdout=f, stderr=f)
            
            subprocess.run(cmd_args+['--RS'], stdout=f, stderr=f)       # 包含了 '-opt', '3'
            
            # subprocess.run(cmd_args+['-opt', '1', '-n', f'{iter}_1'], stdout=f, stderr=f)

            # subprocess.run(cmd_args+['--GM', '-opt', '1', '-n', f'{iter}_1g'], stdout=f, stderr=f)
            
            # subprocess.run(cmd_args+['--BM', '-opt', '1', '-n', f'{iter}_1m'], stdout=f, stderr=f)
            
            subprocess.run(cmd_args+['--GM', '--BM', '-opt', '1', '-n', f'{iter}_1gm'], stdout=f, stderr=f)

            # subprocess.run(cmd_args+['-opt', '2', '-n', f'{iter}_2'], stdout=f, stderr=f)

            # subprocess.run(cmd_args+['--GM', '-opt', '2', '-n', f'{iter}_2g'], stdout=f, stderr=f)
            
            # subprocess.run(cmd_args+['--BM', '-opt', '2', '-n', f'{iter}_2m'], stdout=f, stderr=f)
            
            # subprocess.run(cmd_args+['--GM', '--BM', '-opt', '2', '-n', f'{iter}_2gm'], stdout=f, stderr=f)

            # subprocess.run(cmd_args+['-opt', '3', '-n', f'{iter}_3'], stdout=f, stderr=f)
            
            # subprocess.run(cmd_args+['--GM', '-opt', '3', '-n', f'{iter}_3g'], stdout=f, stderr=f)
            # subprocess.run(cmd_args+['--GM', '-opt', '3'], stdout=f, stderr=f)
            
            # subprocess.run(cmd_args+['--BM', '-opt', '3', '-n', f'{iter}_3m'], stdout=f, stderr=f)
            # subprocess.run(cmd_args+['--BM', '-opt', '3'], stdout=f, stderr=f)
            
            # subprocess.run(cmd_args+['--GM', '--BM', '-opt', '3', '-n', f'{iter}_3gm'], stdout=f, stderr=f)
            # subprocess.run(cmd_args+['--GM', '--BM', '-opt', '3'], stdout=f, stderr=f)

            # subprocess.run(cmd_args+['--ex', '-opt', '3'], stdout=f, stderr=f)

            # f.write("\n"*2)

    # with open(log_file, 'a') as f:
    #         # subprocess.run(cmd_args+['--RS'], stdout=f, stderr=f)       # 包含了 '-opt', '3'
            
    #         # for model in ['mob', 'res', 'vgg']:
    #         # for model in ['res']:
    #             # for t in [5,10,15,30,45,60]:
    #             #     f.write(f"time_limited: {t}s\n")

    #             #     subprocess.run(cmd_args+['--SIMU', '--GM', '--BM', '-opt', '3', '-t', f'{t}', '-m', f'{model}'], stdout=f, stderr=f)

    #             #     f.write("\n"*1)
    #             # f.write("* "*30 + "\n"*2)
    #         # for model in ['mob', 'res', 'vgg', 'alex']:
    #         for model in ['sque', 'r50', 'r50x']:
    #             subprocess.run(cmd_args+['-opt', '0', '-m', f'{model}'], stdout=f, stderr=f)
            
    #             subprocess.run(cmd_args+['--RS', '-m', f'{model}'], stdout=f, stderr=f)       # 包含了 '-opt', '3'

    #             subprocess.run(cmd_args+['--SIMU', '--GM', '--BM', '-opt', '3', '-m', f'{model}'], stdout=f, stderr=f)

    #             f.write("\n"*1)
    #         f.write("* "*30 + "\n"*2)

            

    
