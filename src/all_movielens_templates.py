'''
Pretraining Tasks for MovieLens / Netflix / Douban_Monti Datasets
3 Prompt Families (1, 2, 5) -- Rating, Sequential, Traditional (Direct Recommendation)

These datasets are interaction-heavy but text-light, so Task Families 3 (Explanation)
and 4 (Review) are not applicable. The templates are adapted for movie recommendation
context (using "watch" instead of "purchase", "movie" instead of "product").
'''

all_tasks = {}

# =====================================================
# Task Subgroup 1 -- Rating -- 10 Prompts
# =====================================================

task_subgroup_1 = {}

template = {}

'''
Input template:
Which star rating will user {{user_id}} give item {{item_id}}? (1 being lowest and 5 being highest)


Target template:
{{star_rating}}


Metrics:
Accuracy
'''

template['source'] = "Which star rating will user_{} give item_{} ? ( 1 being lowest and 5 being highest )"
template['target'] = "{}"
template['task'] = "rating"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'item_id']
template['target_argc'] = 1
template['target_argv'] = ['star_rating']
template['id'] = "1-1"

task_subgroup_1["1-1"] = template


template = {}
'''
Input template:
How will user {{user_id}} rate this movie: {{item_title}}? (1 being lowest and 5 being highest)


Target template:
{{star_rating}}


Metrics:
Accuracy
'''
template['source'] = "How will user_{} rate this movie : {} ? ( 1 being lowest and 5 being highest )"
template['target'] = "{}"
template['task'] = "rating"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'item_title']
template['target_argc'] = 1
template['target_argv'] = ['star_rating']
template['id'] = "1-2"

task_subgroup_1["1-2"] = template


template = {}
'''
Input template:
Will user {{user_id}} give item {{item_id}} a {{star_rating}}-star rating? (1 being lowest and 5 being highest)


Target template:
{{answer_choices[label]}} (yes/no)


Metrics:
Accuracy
'''
template['source'] = "Will user_{} give item_{} a {}-star rating ? ( 1 being lowest and 5 being highest )"
template['target'] = "{}"
template['task'] = "rating"
template['source_argc'] = 3
template['source_argv'] = ['user_id', 'item_id', 'star_rating']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "1-3"

task_subgroup_1["1-3"] = template


template = {}
'''
Input template:
Does user {{user_id}} like or dislike item {{item_id}}?


Target template:
{{answer_choices[label]}} (like/dislike) - like (4,5) / dislike (1,2,3)

Metrics:
Accuracy
'''
template['source'] = "Does user_{} like or dislike item_{} ?"
template['target'] = "{}"
template['task'] = "rating"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'item_id']
template['target_argc'] = 1
template['target_argv'] = ['like_dislike']
template['id'] = "1-4"

task_subgroup_1["1-4"] = template


template = {}
'''
Input template:
Predict the user {{user_id}}'s preference on item {{item_id}} ({{item_title}})
-1
-2
-3
-4
-5

Target template:
{{answer_choices[star_rating-1]}}

Metrics:
Accuracy
'''
template['source'] = "Predict the user_{} 's preference on item_{} ( {} ) \n -1 \n -2 \n -3 \n -4 \n -5"
template['target'] = "{}"
template['task'] = "rating"
template['source_argc'] = 3
template['source_argv'] = ['user_id', 'item_id', 'item_title']
template['target_argc'] = 1
template['target_argv'] = ['star_rating']
template['id'] = "1-5"

task_subgroup_1["1-5"] = template


template = {}

'''
Input template:
What star rating do you think {{user_desc}} will give item {{item_id}}? (1 being lowest and 5 being highest)


Target template:
{{star_rating}}


Metrics:
Accuracy
'''

template['source'] = "What star rating do you think {} will give item_{} ? ( 1 being lowest and 5 being highest )"
template['target'] = "{}"
template['task'] = "rating"
template['source_argc'] = 2
template['source_argv'] = ['user_desc', 'item_id']
template['target_argc'] = 1
template['target_argv'] = ['star_rating']
template['id'] = "1-6"

task_subgroup_1["1-6"] = template


template = {}
'''
Input template:
How will {{user_desc}} rate this movie: {{item_title}}? (1 being lowest and 5 being highest)


Target template:
{{star_rating}}


Metrics:
Accuracy
'''
template['source'] = "How will {} rate this movie : {} ? ( 1 being lowest and 5 being highest )"
template['target'] = "{}"
template['task'] = "rating"
template['source_argc'] = 2
template['source_argv'] = ['user_desc', 'item_title']
template['target_argc'] = 1
template['target_argv'] = ['star_rating']
template['id'] = "1-7"

task_subgroup_1["1-7"] = template


template = {}
'''
Input template:
Will {{user_desc}} give a {{star_rating}}-star rating for {{item_title}}? (1 being lowest and 5 being highest)


Target template:
{{answer_choices[label]}} (yes/no)


Metrics:
Accuracy
'''
template['source'] = "Will {} give a {}-star rating for {} ? ( 1 being lowest and 5 being highest )"
template['target'] = "{}"
template['task'] = "rating"
template['source_argc'] = 3
template['source_argv'] = ['user_desc', 'star_rating', 'item_title']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "1-8"

task_subgroup_1["1-8"] = template


template = {}
'''
Input template:
Does {{user_desc}} like or dislike {{item_title}}?


Target template:
{{answer_choices[label]}} (like/dislike) - like (4,5) / dislike (1,2,3)

Metrics:
Accuracy
'''
template['source'] = "Does {} like or dislike {} ?"
template['target'] = "{}"
template['task'] = "rating"
template['source_argc'] = 2
template['source_argv'] = ['user_desc', 'item_title']
template['target_argc'] = 1
template['target_argv'] = ['like_dislike']
template['id'] = "1-9"

task_subgroup_1["1-9"] = template


template = {}
'''
Input template:
Predict {{user_desc}}'s preference towards {{item_title}} (1 being lowest and 5 being highest)

Target template:
{{answer_choices[star_rating-1]}}

Metrics:
Accuracy
'''
template['source'] = "Predict {} 's preference towards {} ( 1 being lowest and 5 being highest )"
template['target'] = "{}"
template['task'] = "rating"
template['source_argc'] = 2
template['source_argv'] = ['user_desc', 'item_title']
template['target_argc'] = 1
template['target_argv'] = ['star_rating']
template['id'] = "1-10"

task_subgroup_1["1-10"] = template


all_tasks['rating'] = task_subgroup_1


# =====================================================
# Task Subgroup 2 -- Sequential -- 13 Prompts
# =====================================================

task_subgroup_2 = {}

template = {}

'''
Input template:
Given the following watching history of user {{user_id}}:
{{history item list of {{item_id}}}}
predict next possible item to be watched by the user?


Target template:
{{item [item_id]}}


Metrics:
HR, NDCG, MRR
'''

template['source'] = "Given the following watching history of user_{} : \n {} \n predict next possible item to be watched by the user ?"
template['target'] = "{}"
template['task'] = "sequential"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'watch_history']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "2-1"

task_subgroup_2["2-1"] = template


template = {}
'''
Input template:
I find the watching history list of user {{user_id}}:
{{history item list of {{item_id}}}}
I wonder which is the next item to recommend to the user. Can you help me decide?


Target template:
{{item [item_id]}}


Metrics:
HR, NDCG, MRR
'''
template['source'] = "I find the watching history list of user_{} : \n {} \n I wonder what is the next item to recommend to the user . Can you help me decide ?"
template['target'] = "{}"
template['task'] = "sequential"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'watch_history']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "2-2"

task_subgroup_2["2-2"] = template


template = {}
'''
Input template:
Here is the watching history list of user {{user_id}}:
{{history item list of {{item_id}}}}
try to recommend next item to the user

Target template:
{{item [item_id]}}


Metrics:
HR, NDCG, MRR
'''
template['source'] = "Here is the watching history list of user_{} : \n {} \n try to recommend next item to the user"
template['target'] = "{}"
template['task'] = "sequential"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'watch_history']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "2-3"

task_subgroup_2["2-3"] = template


template = {}

'''
Input template:
Given the following watching history of {{user_desc}}:
{{history item list of {{item_id}}}}
predict next possible item for the user


Target template:
{{item [item_id]}}


Metrics:
HR, NDCG, MRR
'''

template['source'] = "Given the following watching history of {} : \n {} \n predict next possible item for the user"
template['target'] = "{}"
template['task'] = "sequential"
template['source_argc'] = 2
template['source_argv'] = ['user_desc', 'watch_history']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "2-4"

task_subgroup_2["2-4"] = template


template = {}
'''
Input template:
Based on the watching history of {{user_desc}}:
{{history item list of {{item_id}}}}
Can you decide the next item likely to be watched by the user?


Target template:
{{item [item_id]}}


Metrics:
HR, NDCG, MRR
'''
template['source'] = "Based on the watching history of {} : \n {} \n Can you decide the next item likely to be watched by the user ?"
template['target'] = "{}"
template['task'] = "sequential"
template['source_argc'] = 2
template['source_argv'] = ['user_desc', 'watch_history']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "2-5"

task_subgroup_2["2-5"] = template


template = {}
'''
Input template:
Here is the watching history of {{user_desc}}:
{{history item list of {{item_id}}}}
What to recommend next for the user?

Target template:
{{item [item_id]}}


Metrics:
HR, NDCG, MRR
'''
template['source'] = "Here is the watching history of {} : \n {} \n What to recommend next for the user ?"
template['target'] = "{}"
template['task'] = "sequential"
template['source_argc'] = 2
template['source_argv'] = ['user_desc', 'watch_history']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "2-6"

task_subgroup_2["2-6"] = template


# Extractive QA
template = {}
'''
Input template:
Here is the watching history of user {{user_id}}:
{{history item list of {{item_id}}}}
Select the next possible item likely to be watched by the user from the following candidates:
{{candidate {{item_id}}}}


Target template:
{{item [item_id]}}


Metrics:
HR, NDCG, MRR
'''
template['source'] = "Here is the watching history of user_{} : \n {} \n Select the next possible item likely to be watched by the user from the following candidates : \n {}"
template['target'] = "{}"
template['task'] = "sequential"
template['source_argc'] = 3
template['source_argv'] = ['user_id', 'watch_history', 'candidates']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "2-7"

task_subgroup_2["2-7"] = template


template = {}
'''
Input template:
Given the following watching history of {{user_desc}}:
{{history item list of {{item_id}}}}
What to recommend next for the user? Select one from the following items:
{{candidate {{item_id}}}}

Target template:
{{item [item_id]}}


Metrics:
HR, NDCG, MRR
'''
template['source'] = "Given the following watching history of {} : \n {} \n What to recommend next for the user? Select one from the following items : \n {}"
template['target'] = "{}"
template['task'] = "sequential"
template['source_argc'] = 3
template['source_argv'] = ['user_desc', 'watch_history', 'candidates']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "2-8"

task_subgroup_2["2-8"] = template


template = {}
'''
Input template:
Based on the watching history of user {{user_id}}:
{{history item list of {{item_id}}}}
Choose the next possible item from the following candidates:
{{candidate {{item_id}}}}


Target template:
{{item [item_id]}}


Metrics:
HR, NDCG, MRR
'''
template['source'] = "Based on the watching history of user_{} : \n {} \n Choose the next possible item from the following candidates : \n {}"
template['target'] = "{}"
template['task'] = "sequential"
template['source_argc'] = 3
template['source_argv'] = ['user_id', 'watch_history', 'candidates']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "2-9"

task_subgroup_2["2-9"] = template


template = {}
'''
Input template:
I find the watching history list of {{user_desc}}:
{{history item list of {{item_id}}}}
I wonder which is the next item to recommend to the user. Try to select one from the following candidates:
{{candidate {{item_id}}}}

Target template:
{{item [item_id]}}


Metrics:
HR, NDCG, MRR
'''
template['source'] = "I find the watching history list of {} : \n {} \n I wonder which is the next item to recommend to the user . Try to select one from the following candidates : \n {}"
template['target'] = "{}"
template['task'] = "sequential"
template['source_argc'] = 3
template['source_argv'] = ['user_desc', 'watch_history', 'candidates']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "2-10"

task_subgroup_2["2-10"] = template


# Pairwise Prediction
template = {}
'''
Input template:
User {{user_id}} has the following watching history:
{{history item list of {{item_id}}}}
Does the user likely to watch {{item [item_id]}} next?

Target template:
{{answer_choices[label]}} (yes/no)

Metrics:
Accuracy
'''
template['source'] = "user_{} has the following watching history : \n {} \n does the user likely to watch {} next ?"
template['target'] = "{}"
template['task'] = "sequential"
template['source_argc'] = 3
template['source_argv'] = ['user_id', 'watch_history', 'item_id']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "2-11"

task_subgroup_2["2-11"] = template


template = {}
'''
Input template:
According to {{user_desc}}'s watching history list:
{{history item list of {{item_id}}}}
Predict whether the user will watch {{item [item_id]}} next?

Target template:
{{answer_choices[label]}} (yes/no)

Metrics:
Accuracy
'''
template['source'] = "According to {} 's watching history list : \n {} \n Predict whether the user will watch {} next ?"
template['target'] = "{}"
template['task'] = "sequential"
template['source_argc'] = 3
template['source_argv'] = ['user_desc', 'watch_history', 'item_id']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "2-12"

task_subgroup_2["2-12"] = template


template = {}
'''
Input template:
According to the watching history of {{user_desc}}:
{{history item list of {{item_id}}}}
Can you recommend the next possible item to the user?

Target template:
{{item [item_id]}}


Metrics:
HR, NDCG, MRR
'''
template['source'] = "According to the watching history of {} : \n {} \n Can you recommend the next possible item to the user ?"
template['target'] = "{}"
template['task'] = "sequential"
template['source_argc'] = 2
template['source_argv'] = ['user_desc', 'watch_history']
template['target_argc'] = 1
template['target_argv'] = ['item_id']
template['id'] = "2-13"

task_subgroup_2["2-13"] = template


all_tasks['sequential'] = task_subgroup_2


# =====================================================
# Task Subgroup 5 -- Traditional (Direct) -- 8 Prompts
# =====================================================

task_subgroup_5 = {}

template = {}

'''
Input template:
Will user {{user_id}} likely to interact with item {{item_id}}?


Target template:
{{answer_choices[label]}} (yes/no)


Metrics:
Accuracy (HR, NDCG, MRRs)
'''

template['source'] = "Will user_{} likely to interact with item_{} ?"
template['target'] = "{}"
template['task'] = "traditional"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'item_id']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "5-1"

task_subgroup_5["5-1"] = template


template = {}

'''
Input template:
Shall we recommend item {{item_id}} to {{user_desc}}?


Target template:
{{answer_choices[label]}} (yes/no)


Metrics:
Accuracy (HR, NDCG, MRRs)
'''

template['source'] = "Shall we recommend item_{} to {} ?"
template['target'] = "{}"
template['task'] = "traditional"
template['source_argc'] = 2
template['source_argv'] = ['item_id', 'user_desc']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "5-2"

task_subgroup_5["5-2"] = template


template = {}

'''
Input template:
For {{user_desc}}, do you think it is good to recommend {{item_title}}?


Target template:
{{answer_choices[label]}} (yes/no)


Metrics:
Accuracy (HR, NDCG, MRRs)
'''

template['source'] = "For {}, do you think it is good to recommend {} ?"
template['target'] = "{}"
template['task'] = "traditional"
template['source_argc'] = 2
template['source_argv'] = ['user_desc', 'item_title']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "5-3"

task_subgroup_5["5-3"] = template


template = {}

'''
Input template:
I would like to recommend some movies for user {{user_id}}. Is the following movie a good choice?
{{item_title}}


Target template:
{{answer_choices[label]}} (yes/no)


Metrics:
Accuracy (HR, NDCG, MRRs)
'''

template['source'] = "I would like to recommend some movies for user_{} . Is the following movie a good choice ? \n {}"
template['target'] = "{}"
template['task'] = "traditional"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'item_title']
template['target_argc'] = 1
template['target_argv'] = ['yes_no']
template['id'] = "5-4"

task_subgroup_5["5-4"] = template


template = {}

'''
Input template:
Which item of the following to recommend for {{user_desc}}?
{{candidate {{item_id}}}}


Target template:
{{groundtruth {{item ids}}}}


Metrics:
HR, NDCG, MRR
'''

template['source'] = "Which item of the following to recommend for {} ? \n {}"
template['target'] = "{}"
template['task'] = "traditional"
template['source_argc'] = 2
template['source_argv'] = ['user_desc', 'candidates']
template['target_argc'] = 1
template['target_argv'] = ['groundtruth_item_ids']
template['id'] = "5-5"

task_subgroup_5["5-5"] = template


template = {}

'''
Input template:
Choose the best item from the candidates to recommend for {{user_desc}}?
{{candidate {{item_id}}}}


Target template:
{{groundtruth {{item ids}}}}


Metrics:
HR, NDCG, MRR
'''

template['source'] = "Choose the best item from the candidates to recommend for {} ? \n {}"
template['target'] = "{}"
template['task'] = "traditional"
template['source_argc'] = 2
template['source_argv'] = ['user_desc', 'candidates']
template['target_argc'] = 1
template['target_argv'] = ['groundtruth_item_ids']
template['id'] = "5-6"

task_subgroup_5["5-6"] = template


template = {}

'''
Input template:
Pick the most suitable item from the following list and recommend to user {{user_id}}:
{{candidate {{item_id}}}}


Target template:
{{groundtruth {{item ids}}}}


Metrics:
HR, NDCG, MRR
'''

template['source'] = "Pick the most suitable item from the following list and recommend to user_{} : \n {}"
template['target'] = "{}"
template['task'] = "traditional"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'candidates']
template['target_argc'] = 1
template['target_argv'] = ['groundtruth_item_ids']
template['id'] = "5-7"

task_subgroup_5["5-7"] = template


template = {}

'''
Input template:
We want to make recommendation for user {{user_id}}. Select the best item from these candidates:
{{candidate {{item_id}}}}


Target template:
{{groundtruth {{item ids}}}}


Metrics:
HR, NDCG, MRR
'''

template['source'] = "We want to make recommendation for user_{} .  Select the best item from these candidates : \n {}"
template['target'] = "{}"
template['task'] = "traditional"
template['source_argc'] = 2
template['source_argv'] = ['user_id', 'candidates']
template['target_argc'] = 1
template['target_argv'] = ['groundtruth_item_ids']
template['id'] = "5-8"

task_subgroup_5["5-8"] = template


all_tasks['traditional'] = task_subgroup_5
