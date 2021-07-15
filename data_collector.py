import torch
import torchvision.transforms as T
import car_racing as cr
import numpy as np
from pyglet.window import key


''' Generate and save data from user play. '''
class DataCollector():
    
    def __init__(self, max_nr_observations, n_screens, height, width, device='cpu'):
        
        self.device= device
        
        # The maximum amount of observations
        self.max_nr_observations = max_nr_observations
        
        # The maximum number of items to be stored in memory
        self.capacity = self.max_nr_observations 
        
        # Observation format
        self.n_screens = n_screens
        self.height = height
        self.width = width
        self.obs_shape = (self.height, self.width)
                
        # Data paths
        self.obs_data_path = 'pre_train_data/vae_data_{}.pt'.format(self.max_nr_observations)
        
        # Initialize (empty) memory tensors
        self.obs_mem = torch.empty([self.capacity]+[dim for dim in self.obs_shape], dtype=torch.float32, device=self.device)

        # The number of times new data has been pushed to memory
        self.push_count = 0

        # Preprocessing
        self.preprocess_bw = T.Compose([T.ToPILImage(),
                    T.Grayscale(num_output_channels=1),
                    T.Resize((self.height, self.width)),
                    T.ToTensor()])
       
        
    ''' Convert, resize and preprocess black and white observation '''
    def convert_obs_bw(self, obs):
        # shape observation to original size 96x96 with 3 rgb channels
        obs = obs.reshape(96, 96, 3)
        obs = obs.transpose((2, 1, 0))
        
        # stips of bottom part of the image which contains a black bar with the accumulated reward and control value bars, and makes sure the width is equal size as height
        obs = obs[:, 6:90, int(96*0):int(96 * 0.875)]
        
        # Convert to to float and normalize
        obs = np.ascontiguousarray(obs, dtype=np.float32) / 255
        
        obs = torch.from_numpy(np.flip(obs, axis=0).copy())
        
        obs = self.preprocess_bw(obs)
        
        return obs
    

    def add_to_mem(self, obs):
        # add to tensor
        self.obs_mem[self.position()] = obs 

        # increment push count
        self.push_count += 1
        
        
    def position(self):
        # Returns the next position (index) to which data is pushed
        return self.push_count % self.capacity
        
    
    def saveTorch(self):
        # Save the torch with the observations
        torch.save(self.obs_mem, self.obs_data_path)

        
    def max_observations(self):
        # returns False if the max amount of observations has been made.
        return not self.push_count == self.max_nr_observations
    

    def generate_data(self):
        print("Generating data by having the user play:")
        
        a = np.array([0.0, 0.0, 0.0])
    
        def key_press(k, mod):
            global restart
            if k == 0xFF0D:
                restart = True
            if k == key.LEFT:
                a[0] = -1.0
            if k == key.RIGHT:
                a[0] = +1.0
            if k == key.UP:
                a[1] = +1.0
            if k == key.DOWN:
                a[2] = +0.8  # set 1.0 for wheels to block to zero rotation
    
        def key_release(k, mod):
            if k == key.LEFT and a[0] == -1.0:
                a[0] = 0
            if k == key.RIGHT and a[0] == +1.0:
                a[0] = 0
            if k == key.UP:
                a[1] = 0
            if k == key.DOWN:
                a[2] = 0
    
        env = cr.CarRacing()
        env.render()
        env.viewer.window.on_key_press = key_press
        env.viewer.window.on_key_release = key_release
        record_video = False
        if record_video:
            from gym.wrappers.monitor import Monitor
    
            env = Monitor(env, "/tmp/video-test", force=True)
        isopen = True
        while isopen and self.max_observations():
            env.reset()
            total_reward = 0.0
            steps = 0
            restart = False
            while True and self.max_observations():
                obs = env.render(mode ='state_pixels')
                obs = self.convert_obs_bw(obs)
                self.add_to_mem(obs)
                s, r, done, info = env.step([a[0], a[1], a[2]])
                total_reward += r
                if self.push_count % 200 == 0 or done:
                    print("Percent of data recorded: {}% ({} / {})".format((self.push_count/self.max_nr_observations)*100, self.push_count, self.max_nr_observations))
                steps += 1
                isopen = env.render()
                if done or restart or isopen == False:
                    break
            print("Data points recorded so far:", self.push_count)
            if not self.max_observations():
                self.saveTorch()
        env.close()