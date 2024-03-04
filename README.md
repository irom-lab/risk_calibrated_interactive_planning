# Risk-Calibrated Interactive Planning 

This repo contains the code for Risk-Calibrated Interactive Planning, an algorithm for training robots to interact with humans that have dynamic intent.

Our agent learns an _intent-conditioned_ policy that provides a list of actions that the robot can take depending on what the human intends to do. At inferene time, the robot predicts a set of possible actions that correspond to the highest-confidence human intents. This set is chosen using a small set of learnable parameters and a small dataset of calibration scenarios. 

Depending on the predicted action set, our robot has two choices (i) take the chosen (correct) action of if the predicted action set is a singleton, or (ii) ask the human for help by clarifying their intent. We limit the number of _confidently incorrect_ action selections and the number of help instances using statistical risk calibration. 
