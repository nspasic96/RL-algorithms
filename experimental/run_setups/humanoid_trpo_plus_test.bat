set gym-id=Humanoid-v2
set epochs=500
set epoch_len=2048
set delta=0.1
set delta_final=0.07
set learning_rate_state_value=0.00005
set state_value_network_updates=10
set plus_plus_grad_clip=0.5
set lambd=0.85

set wandb_projet_name=%1

FOR /L %%A IN (%2,1,%3) DO (
	python trpo_final.py --seed %%A --wandb_projet_name %wandb_projet_name% --gym-id %gym-id% --epochs %epochs% --epoch_len %epoch_len% --delta %delta% --delta_final %delta_final% --lambd %lambd% --learning_rate_state_value %learning_rate_state_value% --state_value_network_updates %state_value_network_updates% --render False --plus True --plus_plus True --plus_plus_grad_clip %plus_plus_grad_clip% --wandb_log True
)

