from itertools import chain, combinations
import numpy as np

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)))

class BimanualPlanner:

    def __init__(self):
        self.human_locs = np.array([-1, 0, 1])
        self.bin_locs = np.array([-1, -1, 0, 1, 1])
        self.intents = powerset([1, 2, 3, 4, 5])
        self.intents_list = self.get_intent_index_representation()

    def get_intent_index_representation(self):
        valid_indices = self.intents[1:]
        intents_list = []
        for intents in valid_indices:
            new_intent = np.array([0, 0, 0, 0, 0])
            for intent in intents:
                new_intent[intent-1] = 1
            # if new_intent.sum() != 4:
            intents_list.append(new_intent)
        return intents_list

    def plan(self, human_loc_index):
        human_loc = self.human_locs[human_loc_index]

        plans = {
            "bin_1": [],
            "bin_2": [],
            "bin_3": [],
            "bin_4": [],
            "bin_5": []
        }
        plan_list = []
        for i ,intent in enumerate(self.intents_list):
            intent_index = intent == 1
            valid_bins = self.bin_locs[intent_index]
            dist = 10 * np.ones(5)
            dist[intent_index] = np.abs(human_loc - valid_bins)
            plan = np.argmin(dist)
            plans[f"bin_{plan+1}"].append(i)
            plan_list.append(plan)
        return plans, plan_list

    def get_true_label(self, human_intent):

        plan_list = []
        for i ,intent in enumerate(self.intents_list):
            intent_index = intent == 1
            if all(human_intent == intent):
                return i




if __name__ == "__main__":

    bimanual_planner = BimanualPlanner()
    human_loc_index = 0
    bimanual_planner.plan(human_loc_index)