import argparse
import gym
from gym.envs.registration import register
import numpy as np
import tensorflow as tf
import time

import utils
from networks import StateValueNetwork, PolicyNetwork, SoftQNetwork

register(
	id='HopperBulletEnv-v0',
	entry_point='pybullet_envs.gym_locomotion_envs:HopperBulletEnv',
	max_episode_steps=1000,
	reward_threshold=2500.0
	)


parser = argparse.ArgumentParser(description='SAC agent')

parser.add_argument('--gym-id', type=str, default="HopperBulletEnv-v0",
                   help='the id of the gym environment')
parser.add_argument('--learning-rate-state-value', type=float, default=7e-4,
                   help='the learning rate of the optimizer of state-value function')
parser.add_argument('--learning-rate-policy', type=float, default=7e-4,
                   help='the learning rate of the optimizer of policy')
parser.add_argument('--learning-rate-soft-Q-function', type=float, default=7e-4,
                   help='the learning rate of the optimizer of soft Q function')
parser.add_argument('--seed', type=int, default=1,
                   help='seed of the experiment')
parser.add_argument('--episode-length', type=int, default=2000,
                   help='the maximum length of each episode')
parser.add_argument('--total-timesteps', type=int, default=4000000,
                   help='total timesteps of the experiments')
parser.add_argument('--cuda', type=bool, default=False,
                   help='whether to use CUDA whenever possible')
parser.add_argument('--prod-mode', type=bool, default=False,
                   help='run the script in production mode and use wandb to log outputs')
parser.add_argument('--wandb-project-name', type=str, default="cleanRL",
                   help="the wandb's project name")
parser.add_argument('--gradient-step-freq', type=int, default=1,
                    help='freqeuncy of applying gradient step')
parser.add_argument('--gradient-step-number', type=int, default=1,
                    help='how many gradient steps should be applied')
parser.add_argument('--buffer-size', type=int, default=50000,
                    help='the replay memory buffer size')
parser.add_argument('--gamma', type=float, default=0.99,
                   help='the discount factor gamma')
parser.add_argument('--target-network-frequency', type=int, default=500,
                   help="the timesteps it takes to update the target network")
parser.add_argument('--max-grad-norm', type=float, default=0.5,
                   help='the maximum norm for the gradient clipping')
parser.add_argument('--batch-size', type=int, default=32,
                   help="the batch size of sample from the reply memory")
parser.add_argument('--tau', type=float, default=0.005,
                   help="target smoothing coefficient (default: 0.005)")
parser.add_argument('--alpha', type=float, default=0.2,
                   help="Entropy regularization coefficient.")
parser.add_argument('--squash', type=bool, default=True,
                   help="Whether or not to squash actions with tanh and then multiply with highest possible absolute value.")
args = parser.parse_args()

if not args.seed:
    args.seed = int(time.time())
    
sess = tf.Session()

if args.prod_mode:
    import wandb
    wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, tensorboard=True, config=vars(args), name=experiment_name)
    writer = tf.summary.FileWritter("/tmp/experiment_name", sess.graph)
    wandb.save(os.path.abspath(__file__))
 
env = gym.make(args.gym_id)
np.random.seed(args.seed)
#TODO: maybe add some seed in tensorflow?

env.seed(args.seed)
env.action_space.seed(args.seed)
env.observation_space.seed(args.seed)
inputShape = env.observation_space.shape[0]
outputShape = env.action_space.shape[0]
assert isinstance(env.action_space, gym.spaces.Box), "only continuous action space is supported"       

svnMain = StateValueNetwork(sess, inputShape, args.learning_rate_state_value, suffix="-main") #params are psi
svnAux = StateValueNetwork(sess, inputShape, args.learning_rate_state_value, suffix="-aux") #params are psi hat
pn = PolicyNetwork(sess, inputShape, outputShape, args.learning_rate_policy, args.squash) #params are phi
sqf1 = SoftQNetwork(sess, inputShape, outputShape, args.learning_rate_soft_Q_function, args.alpha, suffix="1") #params are teta
sqf2 = SoftQNetwork(sess, inputShape, outputShape, args.learning_rate_soft_Q_function, args.alpha, suffix="2") #params are teta

init = tf.initialize_all_variables()
init2 = tf.initialize_local_variables()
sess.run(init)
sess.run(init2)

replyBuffer = list()

totalSteps = 0
episodeCount = 0
updateSteps = 0

while totalSteps < args.total_timesteps:
    terminal = False
    stepsCurEpisode = 0
    episodeCount += 1    
    obs = env.reset()
    obs = np.expand_dims(obs, 0)
    
    #episode starts here
    while (args.episode_length > stepsCurEpisode and not terminal):        
        
        actionSquashed, actionRaw, actionMean, actionStd = pn.getSampledAction(obs)
        if args.squash:
            actionTaken = actionSquashed * env.action_space.high
        else:
            actionTaken = actionRaw
            
        #print(actionTaken)
            
        q1 = sqf1.forward(obs, actionTaken)
        q2 = sqf2.forward(obs, actionTaken)        
        minQ = np.minimum(q1[0],q2[0])
        nextObs, reward, terminal, _ = env.step(actionTaken[0])       
        nextObs = np.expand_dims(nextObs, 0)
        stateValueAux = svnAux.forward(nextObs)
        
        instance = (obs, (actionSquashed, actionRaw, actionMean, actionStd), reward, stateValueAux, minQ)
        utils.addToReplyBuffer(replyBuffer, instance, totalSteps, args.buffer_size)
        
        if (totalSteps % args.gradient_step_freq == 0 and len(replyBuffer) >= args.batch_size):
            for i in range(args.gradient_step_number):                
                sampledBatchIdxs = np.random.choice(len(replyBuffer), args.batch_size, replace=False)
                sampledBatch = [replyBuffer[i] for i in sampledBatchIdxs]
                
                observations, logPis, actionsTaken, rewards, minQs, QsHat = utils.prepareInputs(sampledBatch, args.gamma, args.squash)
                                
                svnMain.train(observations, minQs, logPis)
                pn.train(observations, actionsTaken, minQs)
                sqf1.train(observations, actionsTaken, rewards, QsHat, args.gamma)
                sqf2.train(observations, actionsTaken, rewards, QsHat, args.gamma)
                
                svnAux.update(svnMain, args.tau)
                
                updateSteps += 1
                
                svnAuxSync = updateSteps % args.target_network_frequency == 0
                if svnAuxSync:
                    svnAux.weightsAssign(svnMain)                
    
        obs = nextObs
        totalSteps += 1
        stepsCurEpisode += 1
        