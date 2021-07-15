import numpy as np
import datetime
import car_racing as cr

class Agent():
    
    def __init__(self, device = 'cpu'):
        
        self.device = device
        self.env = cr.CarRacing()
        self.render_view = False     # Set to True if you want to see what it is doing
        self.print_timer = 10        # Print average result of Agent every '...' episodes

        self.n_episodes = 1000          # number of episodes
        self.max_length_episode = 1000  # steps per episode
        
        self.run_id = 1
        self.save_results = True
        self.results_path = "results/random/random_results_r{}.npz".format(self.run_id)
 
        self.log_path = "logs/random_log_r{}.txt".format(self.run_id)
        self.record = open(self.log_path, "a")
        self.record.write("\n\n-----------------------------------------------------------------\n")
        self.record.write("File opened at {}\n".format(datetime.datetime.now()))     
        
   
    def train(self):
        
        msg = "Environment is: {}\nTraining started at {}".format("CarRacing-v0", datetime.datetime.now())
        print(msg)
        self.record.write(msg+"\n")
        
        results = []
        for ith_episode in range(self.n_episodes):
            
            # initialize variables
            self.env.reset()
            total_reward = 0
            reward = 0
            nr_steps = 0
            done = False
            
            while not done and nr_steps <= self.max_length_episode:
                
                # select random action
                action = self.env.action_space.sample()
                
                #take step
                _, reward, done, _ = self.env.step(action)
                nr_steps = nr_steps + 1
                
                # render for human if True
                if self.render_view:
                    self.env.render('human')
                
                # add reward to total
                total_reward += reward     
                    
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
        
        #ipythondisplay.clear_output(wait=True)
        self.env.close()
        
        # If enabled, save the results and the network (state_dict)
        if self.save_results:
            np.savez(self.results_path, np.array(results))
       
        # Print and keep a (.txt) record of stuff
        msg = "Training finished at {}".format(datetime.datetime.now())
        print(msg)
        self.record.write(msg)
        self.record.close()
                
if __name__ == "__main__":
    agent = Agent()
    agent.train()