import numpy as np
import matplotlib.pyplot as plt

# Load the data of the DQN
dqn_data1 = np.load("results/dqn/dq_cnn_CarRacing_results_r1.npz")
dqn_data2 = np.load("results/dqn/dq_cnn_CarRacing_results_r2.npz")
dqn_data3 = np.load("results/dqn/dq_cnn_CarRacing_results_r3.npz")
dqn_data4 = np.load("results/dqn/dq_cnn_CarRacing_results_r4.npz")
dqn_data5 = np.load("results/dqn/dq_cnn_CarRacing_results_r5.npz")
dqn_data6 = np.load("results/dqn/dq_cnn_CarRacing_results_r6.npz")
dqn_data7 = np.load("results/dqn/dq_cnn_CarRacing_results_r7.npz")
dqn_data8 = np.load("results/dqn/dq_cnn_CarRacing_results_r8.npz")
dqn_data9 = np.load("results/dqn/dq_cnn_CarRacing_results_r9.npz")
dqn_data10 = np.load("results/dqn/dq_cnn_CarRacing_results_r10.npz")

# Put data in array
dqn_data = np.array([dqn_data1['arr_0'], dqn_data2['arr_0'], dqn_data3['arr_0'], dqn_data4['arr_0'], dqn_data5['arr_0' ],
                      dqn_data6['arr_0'], dqn_data7['arr_0'], dqn_data8['arr_0'], dqn_data9['arr_0'], dqn_data10['arr_0']])
      

# Load the data of the daif agent
daif_data1 = np.load("results/daif/daif_CarRacing_results_r1.npz")
daif_data2 = np.load("results/daif/daif_CarRacing_results_r2.npz")
daif_data3 = np.load("results/daif/daif_CarRacing_results_r3.npz")
daif_data4 = np.load("results/daif/daif_CarRacing_results_r4.npz")
daif_data5 = np.load("results/daif/daif_CarRacing_results_r5.npz")
daif_data6 = np.load("results/daif/daif_CarRacing_results_r6.npz")
daif_data7 = np.load("results/daif/daif_CarRacing_results_r7.npz")
daif_data8 = np.load("results/daif/daif_CarRacing_results_r8.npz")
daif_data9 = np.load("results/daif/daif_CarRacing_results_r9.npz")
daif_data10 = np.load("results/daif/daif_CarRacing_results_r10.npz")

# Put data in array
daif_data = np.array([daif_data1['arr_0'], daif_data2['arr_0'], daif_data3['arr_0'], daif_data4['arr_0'], daif_data5['arr_0'],
                      daif_data6['arr_0'], daif_data7['arr_0'], daif_data8['arr_0'], daif_data9['arr_0'], daif_data10['arr_0']])

daif_data_2 = np.array([daif_data1['arr_0'], daif_data3['arr_0'], daif_data5['arr_0' ],
                       daif_data7['arr_0'], daif_data9['arr_0'], daif_data10['arr_0']])

# Load the data of the random agent
rand_data1 = np.load("results/random/random_results_r1.npz")
rand_data2 = np.load("results/random/random_results_r2.npz")
rand_data3 = np.load("results/random/random_results_r3.npz")
rand_data4 = np.load("results/random/random_results_r4.npz")
rand_data5 = np.load("results/random/random_results_r5.npz")
rand_data6 = np.load("results/random/random_results_r6.npz")
rand_data7 = np.load("results/random/random_results_r7.npz")
rand_data8 = np.load("results/random/random_results_r8.npz")
rand_data9 = np.load("results/random/random_results_r9.npz")
rand_data10 = np.load("results/random/random_results_r10.npz")

# Put data in array
rand_data = np.array([rand_data1['arr_0'], rand_data2['arr_0'], rand_data3['arr_0'], rand_data4['arr_0'], rand_data5['arr_0' ],
                      rand_data6['arr_0'], rand_data7['arr_0'], rand_data8['arr_0'], rand_data9['arr_0'], rand_data10['arr_0']])


# average rewards
dqn_avg_rew = np.load("rewards/dqn_cnn_CarRacing_rewards.npz")
daif_avg_rew = np.load("rewards/daif_CarRacing_rewards.npz")

# Labels
dqn_label = "DQN"
daif_label = "dAIF all runs"
daif_label2 = "dAIF all learning runs"
rand_label ="Random"


def plot_average_reward():
    
    def mean(data):
        mean_data = []
        for i in range(0, len(data['arr_0'])):
            mean_data.append(data['arr_0'][0:i+1].mean())
        return mean_data
    
    mean_data= mean(dqn_avg_rew)
    mean_data2 = mean(daif_avg_rew)


    # plot mean reward
    plt.title("Average reward over episodes")
    plt.ylabel("average reward")
    plt.xlabel("episode")
    plt.plot(mean_data, '#1592e2', alpha = 1, label = 'DQN: {:.2f} +- {:.2f}'.format(np.mean(dqn_avg_rew['arr_0'][0:100]), np.std(dqn_avg_rew['arr_0'][0:100])))
    plt.plot(dqn_avg_rew['arr_0'], '#1592e2',  alpha = 0.3)
    plt.plot(mean_data2, '#E59400', alpha = 1, label = 'dAIF: {:.2f} +- {:.2f}'.format(np.mean(daif_avg_rew['arr_0'][0:100]), np.std(daif_avg_rew['arr_0'][0:100])))
    plt.plot(daif_avg_rew['arr_0'], '#E59400',  alpha = 0.3)
    plt.axvline(x=100, c = "r")
    plt.grid(True, linewidth = 0.1, color = 'black', linestyle = '-')
    plt.legend()
    plt.show()
    
    
def plot_moving_avg():
    def mean(data):
        mean_data = []
        for series in data:
            mean_temp = []
            mean_temp.append(series[0])
            for i in range(1,len(series)):
                mean_temp.append((series[i] * 0.1 ) + (mean_temp[i-1] * 0.9 ))
            mean_data.append(mean_temp)
        return mean_data
    
    def mean2(data):
        mean_data = []
        mean_data.append(data['arr_0'][0])
        for i in range(1,len(data['arr_0'])):
            mean_data.append(( data['arr_0'][i] * 0.1 ) + ( mean_data[i-1] * 0.9 ))
            
        return mean_data
    
    dqn_mean_data = mean(dqn_data)
    dqn_mean = np.mean(dqn_mean_data, axis = 0)
    dqn_std = np.std(dqn_mean_data, axis = 0)
    
    daif_mean_data = mean(daif_data)
    daif_mean = np.mean(daif_mean_data, axis = 0)
    daif_std = np.std(daif_mean_data, axis = 0)
    
    daif2_mean_data = mean(daif_data_2)
    daif2_mean = np.mean(daif2_mean_data, axis = 0)
    daif2_std = np.std(daif2_mean_data, axis = 0)
    
    rand_mean_data = mean(rand_data)
    rand_mean = np.mean(rand_mean_data, axis = 0)
    rand_std = np.std(rand_mean_data, axis = 0)
    
    
    # plot mean reward
    plt.title("Moving Average Reward over episodes")
    plt.ylabel("Average reward")
    plt.xlabel("Episode")
    plt.plot(dqn_mean, "#1f77b4", label = dqn_label, alpha = 1)                                              # DQN Mean
    plt.fill_between(range(0,dqn_mean.shape[0]), dqn_mean-dqn_std, dqn_mean+dqn_std, alpha = 0.4)            # DQN Stdev
    
        
    plt.plot(daif_mean, c = "#ff7f0e", label = daif_label, alpha = 1)                                        # dAIF Mean
    plt.fill_between(range(0,daif_mean.shape[0]), daif_mean-daif_std, daif_mean+daif_std, alpha = 0.4)       # dAIF Stdev
    #plt.plot(mean2(daif_data1), '-', label = "best dAIF", alpha = 0.6)
    
    plt.plot(daif2_mean, c = "#2CA02C", label = daif_label2, alpha = 1)                                      # dAIF Mean
    plt.fill_between(range(0,daif2_mean.shape[0]), daif2_mean-daif2_std, daif2_mean+daif2_std, alpha = 0.4)  # dAIF Stdev
    
    
    plt.plot(rand_mean, c = "r", label = rand_label, alpha = 1)                                              # rand Mean
    plt.fill_between(range(0,rand_mean.shape[0]), rand_mean-rand_std, rand_mean+rand_std, alpha = 0.4)       # rand Stdev
    
    plt.plot(mean2(dqn_data8), '--', label = "best DQN", alpha = 0.6)
    plt.plot(mean2(daif_data1), '--', label = "best dAIF", alpha = 0.6)
    
    plt.grid(True, linewidth = 0.1, color = 'black', linestyle = '-', alpha = 1)
    plt.legend()
    plt.show()


#plot_average_reward()
plot_moving_avg()