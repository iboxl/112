# python eval_run.py -o eval_res_ws  -arch Acc_base -t 360 --WS > test_res_ws.log 2>&1 ;
# python eval_run.py -o eval_r50_base -arch Acc_base -m resnet50 > test_r50_base.log 2>&1 ;
# python eval_run.py -o eval_vgg_base -arch Acc_base -m vgg19bn > test_vgg_base.log 2>&1 ;
# python eval_run.py -o eval_google_base -arch Acc_base -m googlenet > test_google_base.log 2>&1 ;
# python eval_run.py -o eval_alex_base -arch Acc_base -m alexnet > test_alex_base.log 2>&1 ;


# python eval_run.py -o eval_base_2 -t 1200 -arch Acc_base > test_base.log 2>&1
python eval_run.py -o eval_64_16_4 -arch Acc_64_16_4 > test_64_16_4.log 2>&1
python eval_run.py -o eval_64_32_8 -arch Acc_64_32_8 > test_64_32_8.log 2>&1
# python eval_run.py -o eval_halfSize -arch Acc_halfSize > test_halfSize.log 2>&1
# python eval_run.py -o eval_doubleSize -arch Acc_doubleSize > test_doubleSize.log 2>&1
# python eval_run.py -o eval_core4 -arch Acc_core4 > test_core4.log 2>&1
# python eval_run.py -o eval_core16 -arch Acc_core16 > test_core16.log 2>&1
# python eval_run.py -o eval_ws -arch Acc_base > test_ws.log 2>&1