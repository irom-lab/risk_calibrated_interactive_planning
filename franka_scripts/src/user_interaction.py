#!/usr/bin/env python3
import time
import openai
import rospy
import openai
import signal
import random
import numpy as np
import json
from std_msgs.msg import String, Bool
from data.calibrate_dataset import timeout, lm, process_mc_raw, temperature_scaling, mc_gen_prompt_template, mc_score_prompt_template

openai.api_key = "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"

# Auxiliary methods
def colorExtract():
        instruction = input("Please give an positional instruction pertaining to blocks and bowls: ")
        file_path = '/home/allen/catkin_ws/src/franka_irom/data/dataset.txt'
        with open(file_path,'r') as file:
            json_data = file.read()
            qhat = json.loads(json_data)[-1]['qhat']  #loads dataset list and reads off last dict element which stores calibration qhat
        # Get prompt for generating multiple choices
        mc_gen_prompt = mc_gen_prompt_template.replace('{instruction}', instruction).strip()
        _, mc_gen_output = lm(mc_gen_prompt, logit_bias={})
        mc_gen_full, mc_gen_all, add_mc_prefix = process_mc_raw(mc_gen_output)
        # get new prompt
        mc_score_prompt = mc_score_prompt_template.replace('{instruction}', instruction).replace('{mc}', mc_gen_full).strip()
        # call LLM API
        mc_score_response, _ = lm(mc_score_prompt, max_tokens=1, logprobs=5)
        top_logprobs_full = mc_score_response["choices"][0]["logprobs"]["top_logprobs"][0]
        top_tokens = [token.strip() for token in top_logprobs_full.keys()]
        top_logprobs = [value for value in top_logprobs_full.values()]
        # get the softmax value of true option
        mc_smx_all = temperature_scaling(top_logprobs, temperature=5)
        prediction_set = [
                token for token_ind, token in enumerate(top_tokens)
                if mc_smx_all[token_ind] >= 1 - qhat
            ]
        # print
        print('\n===== Multiple choices generated =====')
        print(mc_gen_full)
        print('\n===== Prediction set =====')
        print("Ordered: ",prediction_set)
        if len(prediction_set) == 1:
            print('Singleton set. No help needed!')
            user_chosen_option = prediction_set[0]
        # If the prediction set has more than one option, human help is triggered. 
        # In that case, put the true option as a single capital letter here.
        else:
            print('Set with multiple options. Help needed!')
            user_chosen_option = input("Please pick a letter for the desired action in the prediction set:")
        action_option = mc_gen_all[sorted(top_tokens).index(user_chosen_option)]
        if len(prediction_set) > 1:
            print('Option chosen by user:', action_option)
        else:
            print('Option from prediction set without user intervention', )
        # extract pick and place locations
        if 'not listed here' in action_option or 'do nothing' in action_option or 'block ' not in action_option:
            print('Invalid option! Cannot execute.')
        else:
         # extract pick_obj from mc
            action_option_split = action_option.split()
            pick_obj_attr = action_option_split[action_option_split.index('block')-1]
            pick_obj = pick_obj_attr + ' block'
          # extract target_obj from mc
            target_obj_attr = action_option_split[action_option_split.index('bowl')-1]
            target_obj = target_obj_attr + ' bowl'
          # extract spatial relation from mc
        if 'left' in action_option:
            relation = 'left'
        elif 'right' in action_option:
            relation = 'right'
        elif 'front' in action_option:
            relation = 'front'
        elif 'back' in action_option or 'behind' in action_option:
            relation = 'back'
        else:
         relation = 'in'
        colorString = pick_obj_attr + " " + target_obj_attr
        return colorString
        

        



class UserInteractEnv:
    def __init__(self):
        super(UserInteractEnv, self).__init__()
        self.pub = rospy.Publisher('/color_strings', String, queue_size=10)
        self.rate = rospy.Rate(1)  # Adjust the publication rate as needed
        # listens async in background for flags: checks to see if movement complete
        rospy.Subscriber('/move_flag', Bool, self.__prompt_callback)
        self.prompted = False

    def __prompt_callback(self,msg):
            self.prompted = msg.data # strictly True or False
            
            

    def sendColor(self):
        user_color_input = colorExtract()
        color_string = String(data=user_color_input)
        self.pub.publish(color_string)
        print('sent color')
        

    def run(self):
        while not rospy.is_shutdown():
                if self.prompted:
                    self.sendColor()
                    self.prompted = False
                else:
                    # async background check for call_backs
                    self.rate.sleep()
                
                        

if __name__ == '__main__':
        rospy.init_node('user_interaction_node', anonymous=True)
        time.sleep(4) # allow robot group to initialize
        user_interact_env = UserInteractEnv()   
        user_interact_env.sendColor() # Single initial call
        user_interact_env.run()
