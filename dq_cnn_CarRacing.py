import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torch.optim as optim
import numpy as np
import datetime
import car_racing as cr
import replay_memory as rm

class DQN(nn.Module):
    
    def __init__(self, n_screens, n_hidden, n_outputs, lr=0.001, device='cpu'):
        super(DQN, self).__init__()
        
        self.n_screens = n_screens # Number of stacked observations
        self.n_hidden = n_hidden   # Number of hidden units
        self.n_outputs = n_outputs # Number of outputs

        self.device = device
        
        # The convolutional encoder
        self.encoder = nn.Sequential(                
                nn.Conv2d(self.n_screens, 64, 4, stride=2), # (1, n_screens, 42, 42) --> (1, 64, 20, 20)
                nn.BatchNorm2d(64),
                nn.MaxPool2d(2, stride = 2),                # (1, 64, 20, 20) --> (1, 64, 10, 10)
                nn.ReLU(inplace=False),
                
                nn.Conv2d(64, 128, 4, stride=2),            # (1, 64, 10, 10) --> (1, 128, 4, 4)
                nn.BatchNorm2d(128),
                nn.MaxPool2d(2, stride = 2),                # (1, 128, 4, 4) --> (1, 128, 2, 2)
                nn.ReLU(inplace=False),
                
                nn.Conv2d(128, 256, 2, stride=2),           # (1, 128, 2, 2) --> (1, 256, 1, 1)
                nn.ReLU(inplace=False),
                
                ).to(self.device)
        
        # The size of the encoder output
        self.encoder_output_shape = (256, 1, 1) 
        self.encoder_output_size = np.prod(self.encoder_output_shape)
        
        self.fc1 = nn.Linear(self.encoder_output_size, self.n_hidden) # Hidden layer
        self.fc2 = nn.Linear(self.n_hidden, self.n_outputs)           # Output layer
        
        self.optimizer = optim.Adam(self.parameters(), lr)            # Adam optimizer
        
        self.to(self.device)

    def forward(self, x):
        # cast to device
        x = x.to(self.device)
        
        h1 = self.encoder(x)
        
        h2 = F.relu(self.fc1(h1.view(h1.size(0), -1)))
        
        y  = self.fc2(h2)
        
        return y

    
class Agent():
    
    def __init__(self, device = 'cuda'):
        
        self.device = device
        self.env = cr.CarRacing()
        self.render_view = False     # Set to True if you want to see what it is doing
        self.print_timer = 10        # Print average result of Agent every '...' episodes
        
        # Size of observation
        self.height = self.width = 42 # observation size (height and width)
        self.color = 1                # number of colors
        self.n_screens = 8            # number of observations stacked
        
        self.obs_shape = (self.height, self.width)
        self.obs_size = int(np.prod(self.obs_shape)) # The size of the observation
        self.linear = False                          # True if the input is a vector
        
        # Discretization of continuous action space for CarRacing-v0
        # [0] = steering, [1] = accelerating, [2] = braking
        self.discrete_actions = {0 : np.array([0,0,0]),       # do nothing
                                 1 : np.array([-1,0,0]),      # steer sharp left
                                 2 : np.array([1,0,0]),       # steer sharp right
                                 3 : np.array([-0.5,0,0]),    # steer left
                                 4 : np.array([0.5,0,0]),     # steer right
                                 5 : np.array([0,1,0]),       # accelerate 100%
                                 6 : np.array([0,0.5,0]),     # accelerate 50%
                                 7 : np.array([0,0.25,0]),    # accelerate 25%
                                 8 : np.array([0,0,1]),       # brake 100%
                                 9 : np.array([0,0,0.5]),     # brake 50%
                                 10 : np.array([0,0,0.25])}   # brake 25%
        
        # The number of actions available to the agent
        self.n_actions = len(self.discrete_actions) 
        
        
        # Determines how much the agent explores, decreases linearly over time depending eps_decay to the value of eps_min
        self.eps_max = 0.15
        self.eps_min = 0.05
        self.eps = self.eps_max 
        self.eps_decay = 0.00015
        
        
        self.freeze_cntr = 0           # Keeps track of when to (un)freeze the target network
        self.freeze_period = 50        # How long the network is frozen
        self.batch_size = 250
        
        self.n_hidden = 512            # number of hidden units in the model
        self.lr = 1e-5                 # learning rate
        self.gamma = 0.99              # Discount rate
        self.memory_capacity = 300000  # memory size
        self.n_episodes = 1000         # number of episodes
        self.n_play_episodes = 150     # number of episodes used for average reward test
        self.max_length_episode = 1000 # max number of steps a episode of the CarRacing environment lasts
        
        # Load settings
        self.load_network = False
        self.network_load_path = 'networks/dqn/dq_cnn_CarRacing_policynet_r{}.pth'.format(self.run_id)
        
        # Create networks
        self.policy_net = DQN(n_screens = self.n_screens, n_hidden = self.n_hidden, n_outputs=self.n_actions, lr=self.lr, device=self.device)
        self.target_net = DQN(n_screens = self.n_screens, n_hidden = self.n_hidden, n_outputs=self.n_actions, lr=self.lr, device=self.device)
           
        if self.load_network: # If true: load policy network given a path
            self.policy_net.load_state_dict(torch.load(self.network_load_path, map_location = self.device))
            self.policy_net.eval()
            self.eps = self.eps_min # If we load a model, epsilon is set to its minimum
            print("Succesfully loaded the network")
        
        # Make the target network a copy of the policy network
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Initialize the replay memory
        self.memory = rm.ReplayMemory(self.memory_capacity, self.obs_shape, self.obs_size, self.linear, device=self.device)
        
        # Initialize last observations array
        self.obs_batch = np.array([np.zeros((self.height, self.width), dtype = 'float32') for i in range(self.n_screens)])
    
        # When sampling from memory at index i, obs_indices indicates that we want observations with indices i-obs_indices, works the same for the others
        self.obs_indices = [(self.n_screens+1)-i for i in range(self.n_screens+2)]
        self.action_indices = [1]
        self.reward_indices = [1] # should most likely be [0] instead of [1]
        self.done_indices = [0]
        self.max_n_indices = max(max(self.obs_indices, self.action_indices, self.reward_indices, self.done_indices)) + 1
        
        # Used to pre-process the observations (screens)        
        self.preprocess = T.Compose([T.ToPILImage(),
                    T.Grayscale(num_output_channels=1),
                    T.Resize((self.height, self.width)),
                    T.ToTensor()])
        
        # Training settings
        self.run_id = 1
        self.save_results = True
        self.save_network = True
        self.results_path = "results/dqn/dq_cnn_CarRacing_results_r{}.npz".format(self.run_id)
        self.network_save_path = "networks/dqn/dq_cnn_CarRacing_policynet_r{}.pth".format(self.run_id)
        
        self.log_path = "logs/dq_cnn_CarRacing_log_r{}.txt".format(self.run_id)
        self.record = open(self.log_path, "a")
        self.record.write("\n\n-----------------------------------------------------------------\n")
        self.record.write("File opened at {}\n".format(datetime.datetime.now()))

        
    def select_action(self, obs):
        # exploration
        if np.random.rand() <= self.eps:
            return torch.randint(low=0, high=self.n_actions, size=(1,)).to(self.device)
        # exploitation
        else:
            with torch.no_grad():
                action_values = self.policy_net(obs).to(self.device)
                return torch.tensor([torch.argmax(action_values)], dtype=torch.int64, device=self.device)
            
            
    def convert_observation(self, obs):
        # shape observation to original size 96x96 with 3 rgb channels
        obs = obs.reshape(96, 96, 3)
        obs = obs.transpose((2, 1, 0))
        
        # stips of bottom part of the image which contains a black bar with the accumulated reward and control value bars, and makes sure width is equal to height
        obs = obs[:, 6:90, int(96*0):int(96 * 0.875)]
        
        # grayscale and resize
        obs = self.preprocess(torch.from_numpy(np.flip(obs, axis=0).copy()))      
        
        return obs
    
    
    """ stack the X latest observations into one batch """
    def get_obs_batch(self, obs):
        # add new observation to obs_batch, remove oldest.
        self.obs_batch = np.concatenate((obs.numpy(), self.obs_batch[0:self.n_screens-1]), axis = 0)
        
        # resize to (1, self.n_screens, 84, 84) and convert to torch
        obs_batch2 = torch.from_numpy(np.flip(self.obs_batch, axis=0).copy()).unsqueeze(0).to(self.device)
     
        return obs_batch2
    
    
    def learn(self):
        
        # If there are not enough transitions stored in memory, return
        if self.memory.push_count - self.max_n_indices*2 < self.batch_size:
            return
        
        # After every freeze_period time steps, update the target network
        if self.freeze_cntr % self.freeze_period == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.freeze_cntr += 1
        
        # Retrieve transition data in batches
        all_obs_batch, action_batch, reward_batch, done_batch = self.memory.sample(
                self.obs_indices, self.action_indices, self.reward_indices, self.done_indices, self.max_n_indices, self.batch_size)
  
        # Retrieve a batch of observations for 2 consecutive points in tim
        obs_batch = all_obs_batch[:, 0:self.n_screens, :, :].view(self.batch_size, self.n_screens, self.height, self.width)
        next_obs_batch = all_obs_batch[:, 1:self.n_screens+1, :, :].view(self.batch_size, self.n_screens, self.height, self.width)
        
        # Get the q values and the target values, then determine the loss
        value_batch = self.policy_net(obs_batch).gather(1, action_batch).to(self.device)
        target_out = self.target_net(next_obs_batch).to(self.device)
        target_batch = reward_batch + (1-done_batch) * self.gamma * target_out.max(1)[0].view(self.batch_size, 1)  # (1-done_batch) is used to remove the samples which were gathered when the episode was done
        loss = F.mse_loss(target_batch, value_batch)
        
        self.policy_net.optimizer.zero_grad() # Reset the gradient
        loss.backward() # Compute the gradient
        self.policy_net.optimizer.step() # Perform gradient descent
       
        
    ''' Run a trained model without it learning. '''
    def play(self):
        
        rewards = []
        
        for ith_episode in range(self.n_play_episodes):
            
            total_reward = 0
            nr_steps = 0
            obs = self.env.reset()
            obs = self.convert_observation(obs)
            obs_batch = self.get_obs_batch(obs)
            done = False
            
            while not done and nr_steps <= self.max_length_episode:
                
                # get action
                action = self.select_action(obs_batch)
                
                # get actual action from discrete actions dictionary
                action_todo = self.discrete_actions.get(int(action[0]))
                
                # take step
                obs, reward, done, _ = self.env.step([action_todo[0], action_todo[1], action_todo[2]])
                nr_steps = nr_steps + 1
                obs = self.convert_observation(obs)
                obs_batch = self.get_obs_batch(obs)
                
                # render in visible window if True
                if self.render_view:
                    self.env.render('human')
                
                # add reward to total
                total_reward += reward

            rewards.append(total_reward)
            print("Reward for this episode:", total_reward)
            total_reward = 0         

        self.env.close()
        
        np.savez("rewards/dqn_cnn_CarRacing_rewards", np.array(rewards))
        
        
    def train(self):
        msg = "Environment is: {}\nTraining started at {}".format("CarRacing-v0", datetime.datetime.now())
        print(msg)
        self.record.write(msg+"\n")
        
        results = []
        
        for ith_episode in range(self.n_episodes):
            
            # initialize training variables
            total_reward = 0
            reward = 0
            nr_steps = 0
            done = False
            
            # get first observation
            obs = self.env.reset()
            obs = self.convert_observation(obs)
            obs_batch = self.get_obs_batch(obs)
            
            while not done and nr_steps <= self.max_length_episode:
                
                # get action
                action = self.select_action(obs_batch)
                
                # push to memory
                self.memory.push(obs, action, reward, done)
                
                # get actual action from discrete actions dictionary
                action_todo = self.discrete_actions.get(int(action[0]))
                
                # take step
                obs, reward, done, _ = self.env.step([action_todo[0], action_todo[1], action_todo[2]])
                nr_steps = nr_steps + 1
                
                # get new observation
                obs = self.convert_observation(obs)
                obs_batch = self.get_obs_batch(obs)
                
                # render in visible window if True
                if self.render_view:
                    self.env.render('human')
                
                # add reward to total
                total_reward += reward
                
                # have the networks learn
                self.learn()
                
                # check if episode is done
                if done or nr_steps == self.max_length_episode:
                    self.memory.push(obs, -99, -99, True)
            
            # save result
            results.append(total_reward)
            
            # Update epsilon after each episode
            if self.eps > self.eps_min:
                self.eps = self.eps_max - (ith_episode * self.eps_decay)
            
            
            # Print and keep a (.txt) record of stuff
            if ith_episode > 0 and ith_episode % self.print_timer == 0:
                avg_reward = np.mean(results)
                last_x = np.mean(results[-self.print_timer:])
                msg = "Episodes: {:4d}, eps={:3f}, avg score: {:3.2f}, over last {:d}: {:3.2f}".format(ith_episode, self.eps, avg_reward, self.print_timer, last_x)
                print(msg)
                
                # write to log
                self.record.write(msg+"\n")
                
                # save log
                self.record.close()
                self.record = open(self.log_path, "a")
                
        # close environment after training
        self.env.close()
        
        # If enabled, save the results and the network (state_dict)
        if self.save_results:
            np.savez(self.results_path, np.array(results))
        if self.save_network:
            torch.save(self.policy_net.state_dict(), self.network_save_path)
        
        # Print and keep a (.txt) record of stuff
        msg = "Training finished at {}".format(datetime.datetime.now())
        print(msg)
        self.record.write(msg)
        self.record.close()
                

if __name__ == "__main__":
    agent = Agent()
    agent.train()
