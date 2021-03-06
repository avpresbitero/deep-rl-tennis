import numpy as np
import torch


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def contract(sa):
    """
    Takes in environment state or action list of array, which contains multiple agents action/state information, 
    by concatenating their information, thus removing (but not loosing) the agent dimension in the final output. 
    
    Params
    ======       
            sa (listr) : List of Environment states or actions array, corresponding to each agent
                
    """
    return np.array(sa).reshape(1,-1).squeeze()



def retract(size, num_agents, id_agent, sa, debug=False):
    """
    Reads a batch of Environment states or actions, which have been previously concatened to store 
    multiple agent information into a buffer memory, which is originally not designed to handle multiple 
    agents information(such as in the context of MADDPG)
    
    Params
    ======
            size (int): size of the action space of state spaec to decode
            num_agents (int) : Number of agent in the environment (and for which info hasbeen concatenetaded)
            id_agent (int): index of the agent whose informationis going to be retrieved
            sa (torch.tensor) : Batch of Environment states or actions, each concatenating the info of several 
                                agents (This is sampled from the buffer memmory in the context of MADDPG)
            debug (boolean) : print debug information
    
    """
    
    list_indices  = torch.tensor([ idx for idx in range(id_agent * size, id_agent * size + size) ]).to(device)    
    out = sa.index_select(1, list_indices)
   
    if (debug):
        print("\nDebug decode:\n size=",size, " num_agents=", num_agents, " id_agent=", id_agent, "\n")
        print("input:\n", sa,"\n output:\n",out,"\n\n\n")
    return out