import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import datetime
import matplotlib.pyplot as plt
import torchvision.transforms as T
import car_racing as cr
import replay_memory as rm
import data_collector as dc
import random

class VAE(nn.Module):
    # In part taken from:
    # https://github.com/pytorch/examples/blob/master/vae/main.py

    def __init__(self, n_screens, n_latent_states, lr=1e-5, device='cpu'):
        super(VAE, self).__init__()
        
        self.device = device
        
        self.n_screens = n_screens
        self.n_latent_states = n_latent_states
        
        # The convolutional encoder
        self.encoder = nn.Sequential(                
                nn.Conv2d(self.n_screens, 32, 4, 2), # (1, 8, 42, 42) --> (1, 32, 20, 20)
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                
                nn.Conv2d(32, 64, 4, 2), # (1, 32, 20, 20) --> (1, 64, 9, 9)
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                
                nn.Conv2d(64, 128, 5, 2), # (1, 64, 9, 9) --> (1, 128, 3, 3)
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                
                nn.Conv2d(128, 256, 3, 2), # (1, 128, 3, 3) --> (1, 256, 1, 1)
                nn.ReLU(inplace=True),
                
                ).to(self.device)
        
        # The size of the encoder output
        self.encoder_output_shape = (256, 1, 1)
        self.encoder_output_size = np.prod(self.encoder_output_shape)
        
        # The convolutional decoder
        self.decoder = nn.Sequential(
                nn.ConvTranspose2d(256, 128, 3, 2), # (1, 256, 1, 1) --> (1, 128, 3, 3)
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                
                nn.ConvTranspose2d(128, 64, 5, 2), # (1, 128, 3, 3) --> (1, 64, 9, 9)
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                
                nn.ConvTranspose2d(64, 32, 4, 2), # (1, 64, 9, 9) --> (1, 32, 20, 20)
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                
                nn.ConvTranspose2d(32, self.n_screens, 4, 2), # (1, 32, 20, 20) --> (1, n_screens, 42, 42)
                nn.BatchNorm2d(self.n_screens),
                nn.ReLU(inplace=True),
                
                nn.Sigmoid()
                ).to(self.device)
        
        # Fully connected layers connected to encoder
        self.fc1 = nn.Linear(self.encoder_output_size, self.encoder_output_size // 2) # 1024 --> 512
        self.fc2_mu = nn.Linear(self.encoder_output_size // 2, self.n_latent_states) # 512 --> 128
        self.fc2_logvar = nn.Linear(self.encoder_output_size // 2, self.n_latent_states) # 512 --> 128
        
        # Fully connected layers connected to decoder
        self.fc3 = nn.Linear(self.n_latent_states, self.encoder_output_size // 2) # 128 --> 512
        self.fc4 = nn.Linear(self.encoder_output_size // 2, self.encoder_output_size) # 512 --> 1024
        
        self.optimizer = optim.Adam(self.parameters(), lr)
        
        self.to(self.device)

    def encode(self, x):
        # Deconstruct input x into a distribution over latent states
        conv = self.encoder(x)
        h1 = F.relu(self.fc1(conv.view(conv.size(0), -1)))
        mu, logvar = self.fc2_mu(h1), self.fc2_logvar(h1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        # Apply reparameterization trick
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, batch_size=1):
        # Reconstruct original input x from the (reparameterized) latent states
        h3 = F.relu(self.fc3(z))
        deconv_input = self.fc4(h3)
        deconv_input = deconv_input.view([batch_size] + [dim for dim in self.encoder_output_shape])
        y = self.decoder(deconv_input)
        return y

    def forward(self, x, batch_size=1):
        # Deconstruct and then reconstruct input x
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, batch_size)
        return recon, mu, logvar

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, recon_x, x, mu, logvar, batch=True):
        if batch:
            BCE = F.binary_cross_entropy(recon_x, x, reduction='none')
            BCE = torch.sum(BCE, dim=(1, 2, 3))
            
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        else:
            BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
            # see Appendix B from VAE paper:
            # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
            # https://arxiv.org/abs/1312.6114
            # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return BCE + KLD

class Model(nn.Module):
    
   def __init__(self, n_inputs, n_outputs, n_hidden=64, lr=1e-3, softmax = False, device='cpu'):
        super(Model, self).__init__()
        
        self.n_inputs = n_inputs   # Number of inputs
        self.n_hidden = n_hidden   # Number of hidden units
        self.n_outputs = n_outputs # Number of outputs
        self.softmax = softmax
        
        self.fc1 = nn.Linear(self.n_inputs, self.n_hidden)  # Hidden layer
        self.fc2 = nn.Linear(self.n_hidden, self.n_outputs) # Output layer
        
        self.optimizer = optim.Adam(self.parameters(), lr)  # Adam optimizer
        
        self.device = device
        self.to(self.device)
        
   def forward(self, x):
        # Define the forward pass:
        x = x.to(self.device)
        h1 = F.relu(self.fc1(x))
        
        if self.softmax: # If true apply a softmax function to the output
            y = F.softmax(self.fc2(h1), dim=-1).clamp(min=1e-9, max=1-1e-9) # This is used to get a proper distribution over all actions with a sum of 1.
        else:
            y = self.fc2(h1)
            
        return y
    
class Agent():
    
    def __init__(self, device = 'cuda'):
            
        self.run_id = 1    
        self.device = device
        self.env = cr.CarRacing()
        self.render_view = False      # Set to True if you want to see what it is doing
        self.print_timer = 10         # Print average result of Agent every '...' episodes
        
        self.height = self.width = 42 # observation size (height and width)
        self.color = 1                # number of colors
        self.n_screens = 8            # number of observations stacked
        
        self.obs_shape = (self.height, self.width)  
        self.obs_size = int(np.prod(self.obs_shape))     # The size of the observation
        self.linear = False                              # True if the input is a vector

        # Initialize last observations array
        self.obs_batch = np.array([np.zeros((self.height, self.width), dtype = 'float32') for i in range(self.n_screens)])
        
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
        
        
        self.freeze_cntr = 0           # Keeps track of when to (un)freeze the target network
        self.freeze_period = 50        # How long the network is frozen
        self.batch_size = 250
        self.freeze_vae = True
        
        self.memory_capacity = 100000     # memory size
        self.VAE_memory_capacity = 10000  # VAE pre-train memory size
        self.n_episodes = 1000            # number of episodes
        self.n_play_episodes = 150        # number of episodes used for average reward test
        self.max_length_episode = 1000    # max number of steps a episode of the CarRacing environment lasts
        
        self.gamma = 12                   # Precision parameter
        self.Beta = 0.99                  # Discount factor
        self.alpha = 18000                # VAE loss scaler

        self.n_hidden_trans = 512         # number of hidden units transition network
        self.lr_trans = 1e-3              # learning rate transition network
        self.n_hidden_pol = 512           # number of hidden units policy network
        self.lr_pol = 1e-4                # learning rate policy network
        self.n_hidden_val = 512           # number of hidden units value network
        self.lr_val = 1e-5                # learning rate value network
        
        self.n_latent_states = 128        # size latent space
        self.lr_vae = 5e-6                # learning rate VAE
        self.vae_data = 'pre_train_data/vae_data_10000.pt'
        self.vae_plot = False
        self.pre_train_vae = True         # if True pre-trains the VAE
        
        self.load_pre_trained_vae = True
        self.pt_vae_load_path = "networks/pre_trained_vae/vae_daif_CarRacing_{}_end.pth".format(self.n_latent_states)
        
        self.load_network = False
        self.network_load_path = "networks/daif/daif_CarRacing_{}net_r{}.pth".format("{}", self.run_id)
        
        # Initialize the networks:
        self.vae = VAE(self.n_screens, self.n_latent_states, lr = self.lr_vae, device=self.device)
        self.transition_net = Model(self.n_latent_states*2 + 1, self.n_latent_states, self.n_hidden_trans, lr=self.lr_trans, device=self.device)  # + 1, for 1 action
        self.policy_net = Model(self.n_latent_states*2, self.n_actions, self.n_hidden_pol, lr=self.lr_pol, softmax=True, device=self.device)
        self.value_net = Model(self.n_latent_states*2, self.n_actions, self.n_hidden_val, lr=self.lr_val, device=self.device)
        self.target_net = Model(self.n_latent_states*2, self.n_actions, self.n_hidden_val, lr=self.lr_val, device=self.device)
        self.target_net.load_state_dict(self.value_net.state_dict())
        
        if self.load_pre_trained_vae: # If true: load a pre-trained VAE
            self.vae.load_state_dict(torch.load(self.pt_vae_load_path, map_location=self.device))
            self.vae.eval()
            print("Succesfully loaded a pre-trained VAE")
        
        if self.load_network: # If true: load the networks given paths
            self.vae.load_state_dict(torch.load(self.network_load_path.format("vae"), map_location=self.device))
            self.vae.eval()
            self.transition_net.load_state_dict(torch.load(self.network_load_path.format("trans"), map_location=self.device))
            self.transition_net.eval()
            self.policy_net.load_state_dict(torch.load(self.network_load_path.format("pol"), map_location=self.device))
            self.policy_net.eval()
            self.value_net.load_state_dict(torch.load(self.network_load_path.format("val"), map_location=self.device))
            self.value_net.eval()
            print("Succesfully loaded networks")
            
        
        # Initialize the replay memory
        self.memory = rm.ReplayMemory(self.memory_capacity, self.obs_shape, self.obs_size, self.linear, device=self.device)
        if self.pre_train_vae:
            self.VAE_memory = rm.ReplayMemory(self.VAE_memory_capacity, self.obs_shape, self.obs_size, self.linear, device=self.device)
        
        # When sampling from memory at index i, obs_indices indicates that we want observations with indices i-obs_indices, works the same for the others
        self.obs_indices = [(self.n_screens+1)-i for i in range(self.n_screens+2)]
        self.action_indices = [2, 1]
        self.reward_indices = [1]
        self.done_indices = [0]
        self.max_n_indices = max(max(self.obs_indices, self.action_indices, self.reward_indices, self.done_indices)) + 1
        
        # Used to pre-process the observations (screens)        
        self.preprocess = T.Compose([T.ToPILImage(),
                    T.Grayscale(num_output_channels=1),
                    T.Resize((self.height, self.width)),
                    T.ToTensor()])
        
        self.save_results = True
        self.save_network = True
        self.results_path = "results/daif/daif_CarRacing_results_r{}.npz".format(self.run_id)
        self.network_save_path = "networks/daif/daif_CarRacing_{}net_r{}.pth".format("{}",self.run_id)
                
        
        self.log_path = "logs/daif_CarRacing_log_r{}.txt".format(self.run_id)
        self.record = open(self.log_path, "a")
        self.record.write("\n\n-----------------------------------------------------------------\n")
        self.record.write("File opened at {}\n".format(datetime.datetime.now()))     
        
        
    def get_screen(self, device='cuda'):
        # Get observation, reshape and but in right order.
        screen = self.env.render(mode='state_pixels')        
        screen = screen.reshape(96, 96, 3)
        screen = screen.transpose((2, 1, 0))
        
        # stips of bottom part of the image which contains a black bar with the accumulated reward and control value bars, and makes sure the width is equal size as height
        screen = screen[:, 6:90, int(96*0):int(96 * 0.875)]
        
        # Convert to to float and normalize
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        
        # Add resize
        screen = self.preprocess(torch.from_numpy(np.flip(screen, axis=0).copy()))
        
        return screen

    """ stack the X latest observations into one batch """
    def get_obs_batch(self, obs):
        # add new observation to obs_batch, remove oldest.
        self.obs_batch = np.concatenate((obs.numpy(), self.obs_batch[0:self.n_screens-1]), axis = 0)
        
        # resize to (1, self.n_screens, 84, 84) and convert to torch
        obs_batch2 = torch.from_numpy(np.flip(self.obs_batch, axis=0).copy()).unsqueeze(0).to(self.device)
     
        return obs_batch2

    def select_action(self, obs):
        with torch.no_grad():
            action_index = 0
            
            if self.memory.push_count < self.batch_size + self.n_screens:
                action_index = random.randint(0, self.n_actions - 1)
            else:
                # Derive a distribution over states state from the last n observations (screens):
                prev_n_obs = self.get_obs_batch(obs)
                state_mu, state_logvar = self.vae.encode(prev_n_obs)
                x = torch.cat((state_mu, torch.exp(state_logvar)), dim=1) # does not work?
                policy = self.policy_net(x)
                action_index = torch.multinomial(policy, 1).item()
            
            return action_index
        
    def get_mini_batches(self):
        # Retrieve transition data in mini batches
        all_obs_batch, all_actions_batch, reward_batch_t1, done_batch_t2 = self.memory.sample(
                self.obs_indices, self.action_indices, self.reward_indices,
                self.done_indices, self.max_n_indices, self.batch_size)
        
        # Retrieve a batch of observations for 3 consecutive points in time
        obs_batch_t0 = all_obs_batch[:, 0:self.n_screens, :, :]
        obs_batch_t1 = all_obs_batch[:, 1:self.n_screens+1, :, :]
        obs_batch_t2 = all_obs_batch[:, 2:self.n_screens+2, :, :]
        
        # Retrieve a batch of distributions over states for 3 consecutive points in time
        state_mu_batch_t0, state_logvar_batch_t0 = self.vae.encode(obs_batch_t0)
        state_mu_batch_t1, state_logvar_batch_t1 = self.vae.encode(obs_batch_t1)
        state_mu_batch_t2, state_logvar_batch_t2 = self.vae.encode(obs_batch_t2)
        
        # Combine the sufficient statistics (mean and variance) into a single vector
        state_batch_t0 = torch.cat((state_mu_batch_t0, torch.exp(state_logvar_batch_t0)), dim=1)
        state_batch_t1 = torch.cat((state_mu_batch_t1, torch.exp(state_logvar_batch_t1)), dim=1)
        state_batch_t2 = torch.cat((state_mu_batch_t2, torch.exp(state_logvar_batch_t2)), dim=1)
        
        # Reparameterize the distribution over states for time t1
        z_batch_t1 = self.vae.reparameterize(state_mu_batch_t1, state_logvar_batch_t1)
        
        # Retrieve the agent's action history for time t0 and time t1
        action_batch_t0 = all_actions_batch[:, 0].unsqueeze(1)
        action_batch_t1 = all_actions_batch[:, 1].unsqueeze(1)
        
        # At time t0 predict the state at time t1:
        X = torch.cat((state_batch_t0.detach(), action_batch_t0.float()), dim=1)
        pred_batch_t0t1 = self.transition_net(X)

        # Determine the prediction error wrt time t0-t1:
        pred_error_batch_t0t1 = torch.mean(F.mse_loss(
                pred_batch_t0t1, state_mu_batch_t1, reduction='none'), dim=1).unsqueeze(1)
        
        return (state_batch_t1, state_batch_t2, action_batch_t1,
                reward_batch_t1, done_batch_t2, pred_error_batch_t0t1,
                obs_batch_t1, state_mu_batch_t1,
                state_logvar_batch_t1, z_batch_t1)
        
    def compute_value_net_loss(self, state_batch_t1, state_batch_t2,
                           action_batch_t1, reward_batch_t1,
                           done_batch_t2, pred_error_batch_t0t1):
    
        with torch.no_grad():
            # Determine the action distribution for time t2:
            policy_batch_t2 = self.policy_net(state_batch_t2)
            
            # Determine the target EFEs for time t2:
            target_EFEs_batch_t2 = self.target_net(state_batch_t2)
            
            # Weigh the target EFEs according to the action distribution:
            weighted_targets = ((1-done_batch_t2) * policy_batch_t2 *
                                target_EFEs_batch_t2).sum(-1).unsqueeze(1)
            
            # Determine the batch of bootstrapped estimates of the EFEs:
            EFE_estimate_batch = -reward_batch_t1 + pred_error_batch_t0t1 + self.Beta * weighted_targets
        
        # Determine the EFE at time t1 according to the value network:
        EFE_batch_t1 = self.value_net(state_batch_t1).gather(1, action_batch_t1)

        # Determine the MSE loss between the EFE estimates and the value network output:
        value_net_loss = F.mse_loss(EFE_estimate_batch, EFE_batch_t1)
        
        return value_net_loss
    
    def compute_VFE(self, vae_loss, state_batch_t1, pred_error_batch_t0t1):
        # Determine the action distribution for time t1:
        policy_batch_t1 = self.policy_net(state_batch_t1)
            
        # Determine the EFEs for time t1:
        EFEs_batch_t1 = self.value_net(state_batch_t1)

        # Take a gamma-weighted Boltzmann distribution over the EFEs:
        boltzmann_EFEs_batch_t1 = torch.softmax(-self.gamma * EFEs_batch_t1, dim=1).clamp(min=1e-9, max=1-1e-9)
        
        # Weigh them according to the action distribution:
        energy_term_batch = -(policy_batch_t1 * torch.log(boltzmann_EFEs_batch_t1)).sum(-1).unsqueeze(1)
        
        # Determine the entropy of the action distribution
        entropy_batch = -(policy_batch_t1 * torch.log(policy_batch_t1)).sum(-1).unsqueeze(1)
        
        # Determine the VFE, then take the mean over all batch samples:
        VFE_batch = vae_loss + pred_error_batch_t0t1 + (energy_term_batch - entropy_batch)
        VFE = torch.mean(VFE_batch)
        
        return VFE
    
    def learn(self, ith_episode):
        
        # If there are not enough transitions stored in memory, return
        if self.memory.push_count - self.max_n_indices*2 < self.batch_size:
            return
        
        # After every freeze_period time steps, update the target network
        if self.freeze_cntr % self.freeze_period == 0:
            self.target_net.load_state_dict(self.value_net.state_dict())
        self.freeze_cntr += 1
        
        # Retrieve mini-batches of data from memory
        (state_batch_t1, state_batch_t2, action_batch_t1,
        reward_batch_t1, done_batch_t2, pred_error_batch_t0t1,
        obs_batch_t1, state_mu_batch_t1,
        state_logvar_batch_t1, z_batch_t1) = self.get_mini_batches()
        
        # Determine the reconstruction loss for time t1        
        recon_batch = self.vae.decode(z_batch_t1, self.batch_size)
        
        # Determine the VAE loss for time t1
        vae_loss = self.vae.loss_function(recon_batch, obs_batch_t1, state_mu_batch_t1, state_logvar_batch_t1, batch=True) / self.alpha
        
        # Compute the value network loss:
        value_net_loss = self.compute_value_net_loss(state_batch_t1, state_batch_t2,
                           action_batch_t1, reward_batch_t1,
                           done_batch_t2, pred_error_batch_t0t1)
        
        # Compute the variational free energy:
        VFE = self.compute_VFE(vae_loss, state_batch_t1.detach(), pred_error_batch_t0t1)

        # Reset the gradients:
        if not self.freeze_vae:
            self.vae.optimizer.zero_grad()
        self.policy_net.optimizer.zero_grad()
        self.transition_net.optimizer.zero_grad()
        self.value_net.optimizer.zero_grad()
        
        # Compute the gradients:
        VFE.backward(retain_graph=True)
        value_net_loss.backward()
        
        # Perform gradient descent:
        if not self.freeze_vae:
            self.vae.optimizer.step()
        self.policy_net.optimizer.step()
        self.transition_net.optimizer.step()
        self.value_net.optimizer.step()
        
    ''' Run a trained model without it learning. '''
    def play(self):
        
        rewards = []
        
        self.memory.push_count = self.memory_capacity - 1
        
        for ith_episode in range(self.n_play_episodes):
            
            total_reward = 0
            nr_steps = 0
            obs = self.env.reset()
            obs = self.get_screen(self.device)
            done = False
            
            while not done and nr_steps <= self.max_length_episode:
                
                # get action
                action = self.select_action(obs)
                
                # get actual action from discrete actions dictionary
                action_todo = self.discrete_actions.get(int(action))
                
                # take step
                obs, reward, done, _ = self.env.step([action_todo[0], action_todo[1], action_todo[2]])
                nr_steps = nr_steps + 1
                obs = self.get_screen(self.device)
                
                # render in visible window if True
                if self.render_view:
                    self.env.render('human')
                
                # add reward to total
                total_reward += reward

            rewards.append(total_reward)
            print("Reward for this episode:", total_reward)
            total_reward = 0         

        self.env.close()
        
        np.savez("rewards/daif_CarRacing_rewards", np.array(rewards))
        
        
    def train_vae(self):
        """ Train the VAE using data collected via user play. """
        vae_batch_size = 256
        vae_obs_indices = [self.n_screens-i for i in range(self.n_screens)]
        
        self.VAE_memory.push_count = vae_batch_size + self.n_screens*2
        
        try:
            # Load the pre-collected data into device
            self.VAE_memory.obs_mem = torch.load(self.vae_data, map_location = torch.device(self.device))
        except:
            # Generate data to train VAE on 
            print("No data found to train the vae on.")
            data_collector = dc.DataCollector(self.VAE_memory_capacity, self.n_screens, self.height, self.width, device = self.device)
            data_collector.generate_data()
            self.VAE_memory.obs_mem = torch.load(data_collector.obs_data_path, map_location = torch.device(self.device))
        
        losses = []
        
        for data_point in range(0, len(self.VAE_memory.obs_mem)):
            self.VAE_memory.push_count = self.VAE_memory.push_count + 1
            
            obs_batch, _, _, _ = self.VAE_memory.sample(vae_obs_indices, [], [], [], len(vae_obs_indices), vae_batch_size)
            obs_batch = obs_batch.view(vae_batch_size, self.n_screens, self.height, self.width)
            
            recon, mu, logvar = self.vae.forward(obs_batch, vae_batch_size)
            loss = torch.mean(self.vae.loss_function(recon, obs_batch, mu, logvar))
                    
            self.vae.optimizer.zero_grad()
            loss.backward()
            self.vae.optimizer.step()
                    
            losses.append(loss)
            if data_point % 50 == 0:
                print("obs: %2f vae loss=%5.2f"%(data_point, loss.item()))
                    
            if data_point % 1000 == 0 and data_point > 0 and self.vae_plot:
                plt.plot(losses)
                plt.show()
                plt.plot(losses[-1000:])
                plt.show()
                
                for i in range(self.n_screens): 
                    plt.imsave("vae_images/obs/vae_obs_CarRacing_ep_{}_{}.png".format(data_point, i), obs_batch[0, i, :, :].detach().cpu().squeeze(0).permute(1, 0).numpy(), cmap='gray')
                    plt.imsave("vae_images/recon/vae_recon_CarRacin_ep_{}_{}.png".format(data_point, i), recon[0, i, :, :].detach().cpu().squeeze(0).permute(1, 0).numpy(), cmap='gray')
                    
        self.VAE_memory.push_count = 0
        torch.save(self.vae.state_dict(), "networks/pre_trained_vae/vae_daif_CarRacing_{}_test_end.pth".format(self.n_latent_states))
        
        
    def train(self):

        if self.pre_train_vae and not self.load_pre_trained_vae: # If True: pre-train the VAE
            msg = "Environment is: {}\nPre-training vae. Starting at {}".format("CarRacing-v0", datetime.datetime.now())
            print(msg)
            self.record.write(msg+"\n")
            self.train_vae()
        
        
        msg = "Environment is: {}\nTraining started at {}".format("CarRacing-v0", datetime.datetime.now())
        print(msg)
        self.record.write(msg+"\n")
        
        results = []
        for ith_episode in range(self.n_episodes):
            
            total_reward = 0
            self.env.reset()
            obs = self.get_screen(self.device)
            done = False
            reward = 0
            nr_steps = 0
            
            self.prev_screen = self.env.render('rgb_array')
            
            while not done and nr_steps <= self.max_length_episode:
                
                # get action
                action = self.select_action(obs)
                
                # push to memory
                self.memory.push(obs, action, reward, done)
                
                # get actual action input
                action_real = self.discrete_actions.get(int(action))
                    
                # take step
                obs, reward, done, _ = self.env.step([action_real[0], action_real[1], action_real[2]])
                obs = self.get_screen(self.device)
                nr_steps = nr_steps + 1
                
                # render in visible window if True
                if self.render_view:
                    self.env.render('human')
                
                # add reward to total
                total_reward += reward
                
                # have the networks learn
                self.learn(ith_episode)
                
                if done or nr_steps == self.max_length_episode:
                    self.memory.push(obs, -99, -99, True)
                    
            results.append(total_reward)
            
            # Print and keep a (.txt) record of stuff
            if ith_episode > 0 and ith_episode % self.print_timer == 0:
                avg_reward = np.mean(results)
                last_x = np.mean(results[-self.print_timer:])
                msg = "Episodes: {:4d}, avg score: {:3.2f}, over last {:d}: {:3.2f}".format(ith_episode, avg_reward, self.print_timer, last_x)
                print(msg)
                
                # write to log
                self.record.write(msg+"\n")
                    
                # save log
                self.record.close()
                self.record = open(self.log_path, "a")
            
        self.env.close()
        
        # If enabled, save the results and the network (state_dict)
        if self.save_results:
            np.savez(self.results_path, np.array(results))
        if self.save_network:
            torch.save(self.transition_net.state_dict(), self.network_save_path.format("trans"))
            torch.save(self.policy_net.state_dict(), self.network_save_path.format("pol"))
            torch.save(self.value_net.state_dict(), self.network_save_path.format("val"))
            torch.save(self.VAE.state_dict(), self.network_save_path.format("VAE"))
        
        # Print and keep a (.txt) record of stuff
        msg = "Training finished at {}".format(datetime.datetime.now())
        print(msg)
        self.record.write(msg)
        self.record.close()
                
if __name__ == "__main__":
    agent = Agent()
    agent.train()