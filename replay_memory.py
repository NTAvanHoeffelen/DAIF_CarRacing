import torch
import numpy as np

class ReplayMemory():
    
    def __init__(self, capacity, obs_shape, obs_size, linear = False, device='cpu'):
        
        self.device=device
        
        self.capacity = capacity # The maximum number of items to be stored in memory
        
        # Initialize (empty) memory tensors
        if linear:
            self.obs_mem = torch.empty([capacity]+[obs_size], dtype=torch.float32, device=self.device)
        else:
            self.obs_mem = torch.empty([capacity]+[dim for dim in obs_shape], dtype=torch.float32, device=self.device)
        self.action_mem = torch.empty(capacity, dtype=torch.int64, device=self.device)
        self.reward_mem = torch.empty(capacity, dtype=torch.int16, device=self.device)
        self.done_mem = torch.empty(capacity, dtype=torch.int8, device=self.device)
        
        self.push_count = 0 # The number of times new data has been pushed to memory
        
    def push(self, obs, action, reward, done):
        
        # Store data to memory
        self.obs_mem[self.position()] = obs 
        self.action_mem[self.position()] = action
        self.reward_mem[self.position()] = reward
        self.done_mem[self.position()] = done
        
        self.push_count += 1
    
    def position(self):
        # Returns the next position (index) to which data is pushed
        return self.push_count % self.capacity
    
    
    def sample(self, obs_indices, action_indices, reward_indices, done_indices, max_n_indices, batch_size):  
        # Pick indices at random
        end_indices = np.random.choice(min(self.push_count, self.capacity)-max_n_indices*2, batch_size, replace=False) + max_n_indices
        
        # Correct for sampling near the position where data was last pushed
        for i in range(len(end_indices)):
            if end_indices[i] in range(self.position(), self.position()+max_n_indices):
                end_indices[i] += max_n_indices
        
        # Retrieve the specified indices that come before the end_indices
        obs_batch = self.obs_mem[np.array([index-obs_indices for index in end_indices])]
        action_batch = self.action_mem[np.array([index-action_indices for index in end_indices])]
        reward_batch = self.reward_mem[np.array([index-reward_indices for index in end_indices])]
        done_batch = self.done_mem[np.array([index-done_indices for index in end_indices])]
        
        # Correct for sampling over multiple episodes (change some values when a sample has done = True)
        for i in range(len(end_indices)):
            index = end_indices[i]
            for j in range(1, max_n_indices):
                if self.done_mem[index-j]:
                    for k in range(len(obs_indices)):
                        if obs_indices[k] >= j:
                            obs_batch[i, k] = torch.zeros_like(self.obs_mem[0]) 
                    for k in range(len(action_indices)):
                        if action_indices[k] >= j:
                            action_batch[i, k] = torch.zeros_like(self.action_mem[0]) # Assigning action '0' might not be the best solution, perhaps as assigning at random, or adding an action for this specific case would be better
                    for k in range(len(reward_indices)):
                        if reward_indices[k] >= j:
                            reward_batch[i, k] = torch.zeros_like(self.reward_mem[0]) # Reward of 0 will probably not make sense for every environment
                    for k in range(len(done_indices)):
                        if done_indices[k] >= j:
                            done_batch[i, k] = torch.zeros_like(self.done_mem[0]) 
                    break
                
        return obs_batch, action_batch, reward_batch, done_batch
    
    def get_last_n_obs(self, n, obs_shape):
        """ Get the last n observations stored in memory (of a single episode) """
        last_n_obs = torch.zeros([n]+[dim for dim in obs_shape], device=self.device)
        
        n = min(n, self.push_count)
        for i in range(1, n+1):
            if self.position() >= i:
                if self.done_mem[self.position()-i]:
                    return last_n_obs
                last_n_obs[-i] = self.obs_mem[self.position()-i]
            else:
                if self.done_mem[-i+self.position()]:
                    return last_n_obs
                last_n_obs[-i] = self.obs_mem[-i+self.position()]
        
        return last_n_obs