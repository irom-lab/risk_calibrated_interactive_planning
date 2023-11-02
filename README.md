# PredictiveRL

This repo contains the code for PolicyMesh, an algorithm for training robots to interact with humans that have dynamic intent.

At train time, our agent learns a _multi-modal_ policy using a novel variation of PPO. At each time step, the robot interacts with _N_ hypothetical versions of the human and trains the policy corresponding to the human's realized intent. 

At inference time, our agent _predicts_ the intent of the human and computes a set of viable actions. If the set of viable actions is a singleton, the robot executes the action. If the set is not a singleton, the robot asks the human for their true intent. 
