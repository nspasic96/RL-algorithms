set gym-id=Walker2d-v2
set epochs=500
set epoch_len=2048
set learning_rate_state_value=0.0003
set learning_rate_policy=0.0004

set wandb_projet_name=%1

FOR /L %%A IN (%2,1,%3) DO (
	python ppo_final.py --seed %%A --wandb_projet_name %wandb_projet_name% --gym-id %gym-id% --epochs %epochs% --epoch_len %epoch_len% --learning_rate_state_value %learning_rate_state_value% --learning_rate_policy %learning_rate_policy% --render False --wandb_log True
)
