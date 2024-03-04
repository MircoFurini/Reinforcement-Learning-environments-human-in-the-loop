import numpy as np

class FrozenLakeQLearning:
    
    def __init__(self,env,alpha,gamma,epsilon,numberEpisodes):
        import numpy as np
        
        self.env=env
        self.alpha=alpha
        self.gamma=gamma 
        self.epsilon=epsilon 
        self.actionNumber=env.action_space.n 
        self.numberEpisodes=numberEpisodes
        
        # this matrix is the action value function matrix 
        self.q=np.random.uniform(low=0, high=1, size=(env.observation_space.n,self.actionNumber))

        self.rng = np.random.default_rng() #random generator
        
    def simulateEpisodes(self):
        
        for i in range(self.numberEpisodes):
            
            state = self.env.reset()[0]  
            terminated = False      # True when fall in hole or reached goal
            truncated = False       # True when actions > 200
            totalEpisodeReward = 0
            
            while(not terminated and not truncated):
                
                if i < 500: #first 500 episodes random actions
                    action = self.env.action_space.sample() # actions: 0=left,1=down,2=right,3=up
                else:
                    if self.rng.random() < self.epsilon:
                        action = self.env.action_space.sample() # actions: 0=left,1=down,2=right,3=up
                    else:
                        action = np.argmax(self.q[state,:])    
                        
                if i > 7000:
                    self.epsilon=max(self.epsilon-0.001,0)
                       
                new_state,reward,terminated,truncated,_ = self.env.step(action)

                self.q[state,action] = self.q[state,action] + self.alpha * (
                    reward + self.gamma * np.max(self.q[new_state,:]) - self.q[state,action]
                )
                
                totalEpisodeReward += reward
                
                state = new_state

            print(f"Episode: {i}, total reward: {totalEpisodeReward}")

            self.env.reset()

        self.env.close()

    def learnedAction(self, state):
            
        return np.random.choice(np.where(self.q[state]==np.max(self.q[state]))[0])
    
    def getQ(self):
        
        return self.q 