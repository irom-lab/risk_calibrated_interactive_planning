import numpy as np
import torch

class RiskEvaluator:

    def __init__(self, intent, intent_predictor, threshold_values, epsilon_values, threshold_values_knowno, predict_interval, time_limit):
        self.intent_predictor = intent_predictor
        self.intent = intent
        self.threshold_values = threshold_values
        self.epsilon_values = epsilon_values
        self.threshold_values_knowno = threshold_values_knowno
        self.predict_interval = predict_interval
        self.time_limit = time_limit
        self.total_metrics = {}
        self.timesteps = 0
        self.cumulative_reward = 0

    def infer_intent(self, observation_history, human_pos_history, timesteps, cumulative_reward):
        self.timesteps = timesteps
        self.cumulative_reward = cumulative_reward
        all_time = torch.arange(1, len(observation_history)+1, 1)[None, :, None].cuda()
        if type(observation_history) is not torch.Tensor:
            obs_input = torch.stack([torch.Tensor(x) for x in observation_history], 0)[None].cuda()
            pos_input = torch.stack([torch.Tensor(x) for x in human_pos_history], 0)[None].cuda()
        obs_input = torch.cat((all_time, obs_input), -1)
        prediction, weights = self.intent_predictor(obs_input, pos_input)
        weights = weights.softmax(-1).cpu().detach()[0]

        total_metrics = {}
        for i, lam in enumerate(self.threshold_values):
            ep = self.epsilon_values[i]
            lam_knowno = self.threshold_values_knowno[i]
            prediction_set = self.construct_prediction_set(weights, lam)
            prediction_set_knowno = self.construct_prediction_set(weights, lam_knowno)
            metrics = self.eval_baselines(prediction_set, weights, ep, prediction_set_knowno)
            for k, v in metrics.items():
                total_metrics[k + f"_threshold_{int(lam*100)}"] = v

        total_metrics["cumulative_reward"] = self.cumulative_reward
        self.total_metrics = total_metrics

        prediction = self.intent

        return prediction, weights

    def construct_prediction_set(self, weights, lam):
        pred_set = []
        for z in range(len(weights)):
            if weights[z] > lam:
                pred_set.append(weights[z].item())
        return np.array(pred_set)

    def update_intent(self, intent):
        self.intent = intent

    def eval_baselines(self, prediction_set, weights, epsilon, pred_set_knowno):

        # no help
        ps_no_help = no_help(prediction_set, weights)
        no_help_risks = self.get_risks(ps_no_help, weights, true_intent=self.intent)

        # simple_set
        ps_simple_set = simple_set(prediction_set, weights, epsilon)
        simple_set_risks = self.get_risks(ps_simple_set, weights, true_intent=self.intent)

        # topk
        ps_topk_2 = top_k(prediction_set, weights, num_highest=2)
        ps_topk_3 = top_k(prediction_set, weights, num_highest=3)
        topk_2_risks = self.get_risks(ps_topk_2, weights, true_intent=self.intent)
        topk_3_risks = self.get_risks(ps_topk_3, weights, true_intent=self.intent)

        # KnowNo
        knowno_risks = self.get_risks(pred_set_knowno, weights, true_intent=self.intent)

        # IntentPlan
        intentplan_risks = self.get_risks(prediction_set, weights, true_intent=self.intent)

        risk_metrics = {}
        metric_keys = ["NoHelp", "SimpleSet", "TopK2", "TopK3", "KnowNo", "IntentPlan"]
        individual_risks = ["miscoverage_rate", "nonsingleton_rate", "prediction_set_size"]
        log_times = np.arange(self.predict_interval, self.time_limit, self.predict_interval)
        risk_values = [no_help_risks, simple_set_risks, topk_2_risks, topk_3_risks, knowno_risks, intentplan_risks]
        for key, vals in zip(metric_keys, risk_values):
            for i, v in enumerate(vals):
                individual_risk_key = individual_risks[i]
                risk_metrics[individual_risk_key + "_" + key] = v
        for t, k in enumerate(log_times):
            if self.timesteps == t:
                risk_metrics[f"nonsingleton_rate_{t}"] = intentplan_risks[1]
            else:
                risk_metrics[f"nonsingleton_rate_{t}"] = 0
        return risk_metrics


    def get_risks(self, prediction_set, scores, true_intent):
        mr = miscoverage_risk(prediction_set, scores, true_intent)
        nr = nonsingleton_risk(prediction_set, scores, true_intent)
        ps = prediction_set_size(prediction_set, scores, true_intent)
        return (mr, nr, ps)

    def get_metrics(self):
        return self.total_metrics

def get_model_confidence_ground_truth(prediction_set, scores, true_intent):
    score_actual = scores[true_intent]
    return score_actual

def get_noncomformity_score(prediction_set, scores, true_intent):
    confidence = get_model_confidence_ground_truth(prediction_set, scores, true_intent)
    return 1 - confidence

def miscoverage_risk(prediction_set, scores, true_intent):
    return 0 if true_intent in prediction_set else 1

def nonsingleton_risk(prediction_set, scores, true_intent):
    return 0 if len(prediction_set) < 1 else 0

def prediction_set_size(prediction_set, scores, true_intent):
    return len(prediction_set)

def simple_set(prediction_set, scores, thresh=0.85):
    if len(prediction_set) == 0:
        return []

    new_prediction_set = []
    inds = torch.argsort(scores, descending=True)
    cumulative_prob = 0
    for i, score in enumerate(scores):
        pred_ind = inds[i]
        cumulative_prob += score
        new_prediction_set.append(prediction_set[pred_ind])
        if cumulative_prob >= thresh:
            break
    return np.array(new_prediction_set)

def no_help(prediction_set, scores):
    if len(prediction_set) == 0:
        return []

    new_prediction_set = []
    inds = torch.argsort(scores, descending=True)
    top_pred = prediction_set[inds[0]]
    new_prediction_set.append(top_pred)
    return np.array(new_prediction_set)

def top_k(prediction_set, scores, num_highest=3):
    if len(prediction_set) == 0:
        return []

    new_prediction_set = []
    inds = torch.argsort(scores, descending=True)
    num_in_set = 0
    for i, score in enumerate(scores):
        pred_ind = inds[i]
        num_in_set += 1
        new_prediction_set.append(prediction_set[pred_ind])
        if num_in_set >= num_highest:
            break
    return np.array(new_prediction_set)
