important accessory files:

pde_dataset_with_winds.py : dataset class with winds
amse_loss.py : AMSE loss function for phase 3 training
reverse_huber_loss.py : (has bugs)
config_paradis.yaml

forecast codes:

avi_forecast_gbell.py                     -   gbell initial condition in only the geopotential (channel 0); torchharmonics random initial condition otherwise
avi_forecast_gbell_all_channel.py  -   gbell initial condition in all channels
avi_forecast_gbell_burn.py            -  gbell initial condition in only the geopotential with discarding the first few rollout steps so the input to the neural network is few steps after the gaussian bells initial conditions
avi_forecast_gbell_withfooargs.py - forecast code with gbell initial condition in only the geopotential, where neural network architecture specifications can be given as arguments (Important)
avi_forecast_gbell_withfooargs_steadystate_V3.py - Steady state initialization with predicted winds as input for next rollout steps
avi_forecast_gbell_withfooargs_steadystate_V4.py - same as avi_forecast_gbell_withfooargs_steadystate_V3.py but winds are fixed
avi_forecast_gbell_withfooargs_steadystate_and_Rossby_Hurwitz_final.py - steady state initialization + Rossby Hurwitz initialization both possible


Training codes:

Phase 1:
optuna_paradis_sweep_v1.py   -  for optuna training
avi_train_gbell_train_gbell_val_2.py - gbell initial condition only in channel 0 phase 1 training
avi_train_gbell_train_gbell_val_5_all_channel.py   -  all channel gbell initial condition phase 1 training
avi_train_gbell_train_gbell_val_5_all_channel_berhu_ON_2.py   -  all channel phase 1 training with berhu (bugs)

Phase 2:
avi_train_rollout_loss_burnin.py -  only channel 0 (geopotential ) gbell initial condition phase 2
avi_train_rollout_loss_burnin_all_channel.py - all channel initial condition phase 2 training with discard (burn) after a few initial steps starting from gbell initial conditions
avi_train_rollout_loss_burnin_all_channel_berhu.py - same as above but with reverse huber loss function (bugs)
avi_train_rollout_loss_burnin_all_channel_berhu_with_lat_weights.py - same as above but has latitude weights (bugs)

Phase 3:
avi_train_gbell_train_gbell_val_AMSE_Loss_all_channel_burnin.py  - phase 3 training for longer lead times with discarding of initial few steps of gbell initial conditions
avi_train_gbell_train_gbell_val_AMSE_Loss_v2.py   - phase 3 training for 1 step lead time


pbs files that are important :
Phase 1:
              run_3march.pbs
                            run_3march_berhu.pbs
         run_3march_berhu_stable.pbs
                         run_3march_berhu_with_lat_weights.pbs
              run_3march_phase1_stable.pbs

phase 2:
                            run_4march_phase2_stable.pbs
                      run_4march_phase2_stable_berhu.pbs
     run_4march_phase2_stable_berhu_with_weights.pbs

phase 3:
run_4march_phase3_stable.pbs


optuna : 
run_optuna_4gpu.pbs

where the weights are being saved currently:
trial directories
latest:
trial3_3march
trial4_4march
trial2_24feb
trial3_3march_berhu
trial_optuna

how to run a forecast code (example):
$DIFFERENT_PYTHON avi_forecast_gbell_withfooargs_steadystate_and_Rossby_Hurwitz_final.py   --config config_paradis.yaml   --checkpoint trial4_4march/paradis_gbell_train_gbell_val_all_channel_rollout_dt25_train1024_hd48_L8_enc3_vel6_diff24_react12_bias3_seed42_lr0.00005_Burn3_rollout9_finetune_epochs100_4march_rolloutloss/version_0/checkpoints/rolloutloss-epoch=20-val_loss=0.0242.ckpt --output_dir trial3_3march/Phase1output_4march_paradis_40steps_now_best_williamson6_6060606060 --autoreg_steps 40   --device cuda   --plot_channel 0   --seed 31   --tb_logdir logs/Phase1output_4march_paradis_40steps_now_best_output_williamson6_6060606060   --dt_solver 25  --model.paradis.hidden_dim 48   --model.paradis.num_layers 8   --model.paradis.num_encoder_layers 3   --model.paradis.num_vels 6   --model.paradis.diffusion_size 24   --model.paradis.reaction_size 12   --model.paradis.bias_channels 3 --williamson_case6

replace --williamson_case6 (which is for Rosby Hurwitz) with --williamson_case2 (steady state) or with -gbells( for gaussian bells initial conditions)

replace the forecast file with other forecast files based on what each file is capable of according to the above instructions
