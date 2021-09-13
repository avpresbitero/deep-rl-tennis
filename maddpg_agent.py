import numpy as np
import random
import copy
import torch

from ddpg_agent import Agent
from hyperparameter  import *
from utils import contract, retract
from replay import ReplayBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Maddpg():
    """MADDPG Agent : Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, num_agents, random_seed):
        """Initialize a MADDPG Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            num_agents (int): number of agents
            random_seed (int): random seed
        """
        
        super(Maddpg, self).__init__()
        
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = random.seed(random_seed)
        
        # Instantiate Multiple  Agent
        self.agents = [ Agent(state_size,action_size, random_seed, num_agents) 
                       for i in range(num_agents) ]
        
        # Instantiate Replay Buffer that is shared between agents
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        
                  
    def reset(self):
        """Reset all the agents"""
        for agent in self.agents:
            agent.reset()

    def act(self, states, noise):
        """Return action to perform for each agents (per policy)"""        
        return [ agent.act(state, noise) for agent, state in zip(self.agents, states) ]
                
    
    def step(self, states, actions, rewards, next_states, dones, num_current_episode):
        """ Save experience in replay memory, and use random sample from buffer to learn"""
 
        self.memory.add(contract(states), 
                        contract(actions), 
                        rewards,
                        contract(next_states),
                        dones)

        # If enough samples in the replay memory and if it is time to update
        if (len(self.memory) > BATCH_SIZE) and (num_current_episode % UPDATE_EVERY_NB_EPISODE ==0) :
            
            # Note: Expects 2 agents
            assert(len(self.agents)==2)
            
            # Learn several times in an episode
            for i in range(MULTIPLE_LEARN_PER_UPDATE):
                experiences = self.memory.sample()   
                self.maddpg_learn(experiences, self_idx=0, other_idx=1)
                experiences = self.memory.sample()   
                self.maddpg_learn(experiences, self_idx=1, other_idx=0)
                
    
    def maddpg_learn(self, experiences, self_idx, other_idx, gamma=GAMMA):
        """
        Update the policy of the MADDPG agent. The actors have only access to agent's own 
        information, whereas the critics have access to all the agents information.
        
        Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(states) -> action
            critic_target(all_states, all_actions) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            self_idx (int) : index of the own agent to update in self.agents
            other_idx (int) : index of the other agent to update in self.agents
            gamma (float): discount factor
        """
        
        states, actions, rewards, next_states, dones = experiences
               
        self_states =  retract(self.state_size, self.num_agents, self_idx, states)
        self_actions = retract(self.action_size, self.num_agents, self_idx, actions)
        self_next_states = retract(self.state_size, self.num_agents, self_idx, next_states) 
                
        other_states =  retract(self.state_size, self.num_agents, other_idx, states)
        other_actions = retract(self.action_size, self.num_agents, other_idx, actions)
        other_next_states = retract(self.state_size, self.num_agents, other_idx, next_states)
        
        all_states=torch.cat((self_states, other_states), dim=1).to(device)
        all_actions=torch.cat((self_actions, other_actions), dim=1).to(device)
        all_next_states=torch.cat((self_next_states, other_next_states), dim=1).to(device)
   
        agent = self.agents[self_idx]
            
        # ---------------------------- update critic ---------------------------- #
        
        # Get predicted next-state actions and Q values from target models        
        all_next_actions = torch.cat((agent.actor_target(self_states), agent.actor_target(other_states)),
                                     dim =1).to(device) 
        Q_targets_next = agent.critic_target(all_next_states, all_next_actions)
        
        
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        
        # Compute critic loss
        Q_expected = agent.critic_local(all_states, all_actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        if (CLIP_CRITIC_GRADIENT):
            torch.nn.utils.clip_grad_norm(agent.critic_local.parameters(), 1)
        agent.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        all_actions_pred = torch.cat((agent.actor_local(self_states), agent.actor_local(other_states).detach()),
                                     dim = 1).to(device)      
        actor_loss = -agent.critic_local(all_states, all_actions_pred).mean()
        
        # Minimize the loss
        agent.actor_optimizer.zero_grad()
        actor_loss.backward()        
        agent.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        agent.soft_update(agent.critic_local, agent.critic_target, TAU)
        agent.soft_update(agent.actor_local, agent.actor_target, TAU)                   
   
                        
    def checkpoints(self):
        """Save checkpoints for all Agents"""
        for idx, agent in enumerate(self.agents):
            actor_local_filename = 'model/actor_local_' + str(idx) + '.pth'
            critic_local_filename = 'model/critic_local_' + str(idx) + '.pth'           
            actor_target_filename = 'model/actor_target_' + str(idx) + '.pth'
            critic_target_filename = 'model/critic_target_' + str(idx) + '.pth'            
            torch.save(agent.actor_local.state_dict(), actor_local_filename) 
            torch.save(agent.critic_local.state_dict(), critic_local_filename)             
            torch.save(agent.actor_target.state_dict(), actor_target_filename) 
            torch.save(agent.critic_target.state_dict(), critic_target_filename)
            
