import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import random
from random import randint
import time
import pickle

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint

        # Initialize any additional variables here
        # self.q = pickle.load(open('q.p', 'rb'))
        self.q = {}
        self.actions = [None, 'forward', 'left', 'right']
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 5.0
        self.old_state = None
        self.old_action = None
        self.old_reward = 0

        self.i = 0
        self.trials = []
        self.trial_index = 0
        self.first_trial = True

    def reset(self, destination=None):
        """Resets variables for next trial"""
        self.planner.route_to(destination)
        self.state = None
        self.next_waypoint = None
        self.color = 'red'

        self.old_state = None
        self.old_action = None

        # Log each new trial in trials list
        deadline = self.env.get_deadline(self)
        trial = {
            'deadline': deadline,
            'iteration': 0,
            'wrong_moves': 0,
        }
        self.trials.append(trial)
        self.i = 0
        if self.first_trial != True:
            self.trial_index += 1

        self.first_trial = False

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # Update state
        self.state = self.get_agent_state(inputs, self.next_waypoint)

        # Select action according to policy
        action = self.get_action(self.state)

        # Execute action and get reward
        reward = self.env.act(self, action)
        if reward < 0:
            self.trials[self.trial_index]['wrong_moves'] += 1

        # Learn policy based on state, action, reward
        self.learn(self.old_state, self.old_action,
                    self.old_reward, self.state)

        self.old_state = self.state

        self.old_action = action
        self.old_reward = reward
        self.i += 1
        self.trials[self.trial_index]['iteration'] = self.i

        print """LearningAgent.update(): deadline = {}, inputs = {},
                action = {}, reward = {}"""\
                .format(deadline, inputs, action, reward)  # [debug]

    def get_agent_state(self, inputs, next_waypoint):
        """Returns the agents state based on different inputs"""
        state = (inputs['light'], inputs['oncoming'],
                    inputs['left'], next_waypoint)
        return state

    def get_q(self, state, action):
        """Returns the value for a gvien state and action"""
        return self.q.get((state, action), 0) # returns value or defaults to zero

    def learn(self, old_state, old_action, reward, state):
        """
        Q Learning implementation was influenced by Travis DeWolf's
        Cat vsMouse Exploration http://tinyurl.com/zdzace8
        """

        if self.old_state != None:
            old_value = self.get_q(old_state, old_action)

            est_future_value = max([self.get_q(state, i)
                                    for i in self.actions])
            learned_value = reward + self.gamma * est_future_value

            # Update q_matrix
            if old_value == 0:
                self.q[(old_state, old_action)] = reward
            else:
                val = old_value + (self.alpha) * (learned_value - old_value)
                self.q[(old_state, old_action)] = val

    def get_action(self, state):
        """
        Get the action for a given state
        Uses Eplison Greedy to randomize learning during the early trials
        later trials are based on the filled out Q Matrix
        """
        q = [self.get_q(state, i) for i in self.actions]
        max_q = max(q)

        e = 1
        r = random.random()
        if self.trial_index != 0:
            e = self.epsilon / (self.trial_index + 1.)

        # Determine if random action should be taken
        if r < e:
            r = randint(0, 3)
            max_q = q[r]

        count = q.count(max_q)
        if count > 1:
            best = [i for i in range(len(self.actions)) if q[i] == max_q]
            i = random.choice(best)
        else:
            i = q.index(max_q)

        action = self.actions[i]
        return action

def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # set agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0.0000001)  # reduce update_delay to speed up simulation
    sim.run(n_trials=100)  # press Esc or close pygame window to quit

    last_ten = a.trials[-10:]
    wins = 0
    for trial in last_ten:
        net_value = trial['deadline'] - trial['iteration']
        if net_value >= 0:
            wins += 1

    win_percentage = wins / 10.0
    return win_percentage, a.q

if __name__ == '__main__':
    outcome, q = run()
    print 'Results: '
    print outcome

    # print 'Saving q'
    # pickle.dump(q, open('q.p', 'wb'))
