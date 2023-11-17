import torch as T
import torch.nn.functional as F

from networks import ActorNetwork, CriticNetwork


class Agent:
    def __init__(self, actor_dims, critic_dims, n_agents, n_actions, 
                 agent_idx, chkpt_dir,
                 alpha=0.01, beta=0.01, gamma=0.95, tau=0.01, 
                 fc1=64, fc2=64, evaluate=False):
        self.evaluate = evaluate
        self.gamma = gamma
        self.tau = tau
        self.n_actions = n_actions
        self.agent_name = 'agent_%s' % agent_idx
        self.actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions,
                                  self.agent_name+'_actor', chkpt_dir)
        self.critic = CriticNetwork(beta, critic_dims, fc1, fc2, n_agents, n_actions,
                                    self.agent_name+'_critic', chkpt_dir)
        self.target_actor = ActorNetwork(alpha, actor_dims, fc1, fc2, n_actions,
                                         self.agent_name+'_target_actor', chkpt_dir)
        self.target_critic = CriticNetwork(beta, critic_dims, fc1, fc2, n_agents, n_actions,
                                           self.agent_name+'_target_critic', chkpt_dir)

        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        action = self.actor.forward(state)
        if not self.evaluate:
            noise = T.rand(self.n_actions).to(self.actor.device)
            action = action + noise
            action = F.softmax(action, dim=1).to(self.actor.device)

        return action.detach().cpu().numpy()[0]

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_actor_params = self.target_actor.named_parameters()
        actor_params = self.actor.named_parameters()

        target_actor_state_dict = dict(target_actor_params)
        actor_state_dict = dict(actor_params)
        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                (1-tau)*target_actor_state_dict[name].clone()

        self.target_actor.load_state_dict(actor_state_dict)

        target_critic_params = self.target_critic.named_parameters()
        critic_params = self.critic.named_parameters()

        target_critic_state_dict = dict(target_critic_params)
        critic_state_dict = dict(critic_params)
        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                (1-tau)*target_critic_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()
