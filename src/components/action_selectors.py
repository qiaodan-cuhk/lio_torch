import torch as th
from torch.distributions import Categorical
from .epsilon_schedules import DecayThenFlatSchedule

REGISTRY = {}


class MultinomialActionSelector(): #A2C选这个

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)

        self.test_greedy = getattr(args, "test_greedy", True)
        self.save_probs = getattr(self.args, 'save_probs', False)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        masked_policies = agent_inputs.clone()
        masked_policies[avail_actions == 0] = 0


        if test_mode and self.test_greedy:
            picked_actions = masked_policies.max(dim=2)[1]
        else:
            self.epsilon = self.schedule.eval(t_env)

            epsilon_action_num = (avail_actions.sum(-1, keepdim=True) + 1e-8)
            masked_policies = ((1 - self.epsilon) * masked_policies
                        + avail_actions * self.epsilon/epsilon_action_num)
            masked_policies[avail_actions == 0] = 0
            
            picked_actions = Categorical(masked_policies).sample().long()

        if self.save_probs:
            return picked_actions, masked_policies
        else:
            return picked_actions

REGISTRY["multinomial"] = MultinomialActionSelector
