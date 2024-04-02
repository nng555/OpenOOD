# tracin noisy
python3 run_bash.py command="'bash /h/nng/projects/OpenOOD/cifar100_trace.sh cifar100,cpath_\${cpath},trace,train model_epoch\${epoch}.ckpt --dataset.corruption_path \${cpath} --postprocessor.skip_ood True --postprocessor.use_train True'" epoch=20,40,60,80,100,120,140,160,180,200 hydra.launcher.qos=scavenger +hydra.launcher.time=60 tags="[cifar100,tracin_eval]" cpath=human,data,asym_0.3,sym_0.6 -m

# tracin clean
python3 run_bash.py command="'bash /h/nng/projects/OpenOOD/cifar100_trace.sh cifar100,no_noise,res18,trace,train model_epoch\${epoch}.ckpt --postprocessor.skip_ood True --postprocessor.use_train True'" epoch=20,40,60,80,100,120,140,160,180,200 hydra.launcher.qos=scavenger +hydra.launcher.time=60 tags="[cifar100,tracin_eval,no_noise]" -m

# ifcomp noisy
python3 run_bash.py command="'bash /h/nng/projects/OpenOOD/cifar100_trace_comp.sh cifar100,cpath_\${cpath},trace,train --dataset.corruption_path \${cpath} --postprocessor.skip_ood True --postprocessor.use_train True'" hydra.launcher.qos=m2 +hydra.launcher.time=360 tags="[cifar100,ifcomp_eval]" cpath=human,data,asym_0.3,sym_0.6 hydra.launcher.partition=a40 -m

# ifcomp clean
python3 run_bash.py command="'bash /h/nng/projects/OpenOOD/cifar100_trace_comp.sh cifar100,no_noise,res18,trace,train --postprocessor.skip_ood True --postprocessor.use_train True'" hydra.launcher.qos=m2 +hydra.launcher.time=360 tags="[cifar100,ifcomp_eval,no_noise]" hydra.launcher.partition=a40  -m
