import pandas as pd
import numpy as np
from modules import beam_search_models_tensor 
from modules import training_constants as tconst
import torch
from tqdm import tqdm
from heapq import heappush, heappop


def createSelectors(data, no_columns):
     
    selectors = []

    # Dont want to create a selector for the class label
    for i in range(0, no_columns-1):
        sorted, indices = torch.sort(torch.unique(data[: , i]))
        for val in sorted:
            selector = beam_search_models_tensor.EquitySelector(i, val)
            selectors.append(selector)
            
    return selectors



def computeInterestingness(target_share, coverage):
    # return (target_share + tconst.COVERAGE_COEFF*coverage)/tconst.NORMALIZATION_CONST 
    return target_share

def computeMetrics(data, what_sg_covers):

    # Getting some baseline information
    no_columns = len(data[1, :])
    class_index = no_columns - 1
    class_tensor = data[: , class_index]

    # no rows in the dataset 
    no_instances_in_dataset = torch.sum(torch.abs(class_tensor))
    
    # Keep only the rows that are in the subgroup
    sg_data = what_sg_covers.T * data
    
   
    # no rows that are covered by the subgroup
    class_tensor_sg = sg_data[: , class_index]
    no_instances_in_sg = torch.sum(torch.abs(class_tensor_sg))
    
    # If we have an impossible combination of selectors, not interesting at all
    if no_instances_in_sg == 0:
        return 0, 0, 0, 0, 0

    # The no +ve rows in the subgroup
    relu = torch.nn.ReLU()
    result = relu(class_tensor_sg)
    no_positive_instances_in_sg = torch.sum(result)
    

    # Computing the metrics we care about
    target_share = no_positive_instances_in_sg/no_instances_in_sg
    coverage = (no_instances_in_sg/no_instances_in_dataset)  
    interestingness = computeInterestingness(target_share, coverage)

    return interestingness, target_share, coverage, no_positive_instances_in_sg, no_instances_in_sg



# Get metrics for the entire dataset: the null subgroup
def get_info_of_null_sg(data):

    # Getting some baseline information
    no_columns = len(data[1, :])
    class_index = no_columns - 1
    class_tensor = data[: , class_index]

    # no rows in the dataset 
    no_instances = torch.sum(torch.abs(class_tensor))

    # The no +ve rows in the subgroup
    m = torch.nn.ReLU()
    class_tensor_neg_zeroed_out = m(class_tensor)
    no_positive_instances = torch.sum(class_tensor_neg_zeroed_out)

    target_share_null_set = no_positive_instances/no_instances
    coverage_null_set = no_instances/no_instances
    interestingness_null_set = computeInterestingness(target_share_null_set, coverage_null_set)
    
    return 'Empty sg/Entire dataset', interestingness_null_set, target_share_null_set, coverage_null_set, no_positive_instances, no_instances  


def creatingDataframe(beam, sg_null, interestingness_null, target_share_null, coverage_null, no_p_inst_null, no_inst_null, COVERAGE_COEFF, MIN_NUM_INDIVIDUALS_PER_SUBGROUP):

    # Getting all the beam information into a format that I can turn into a dataframe
    prep_for_dataframe = [{ 'subgroup' : s_d.conjunction, 
                            'interestingness' : s_d.interestingness.item(), 
                            'target_share' : s_d.target_share.item(),
                            'coverage' : s_d.coverage.item(), 
                            'no_positive_inst_in_sg' : s_d.no_sg_p_inst.item(),
                            'no_inst_in_sg' : s_d.no_sg_inst.item(),
                            } for s_d in beam]

    # Adding details of the hyperparamters and info of the null subgroup into a list 
    list_to_prepend = [{'subgroup' : 'Details Of Hyperparameters',
                        'interestingness': 'COVERAGE_COEFF: ' + str(COVERAGE_COEFF), 
                        'target_share' : 'MIN NO INDIVIDUALS PER SG: ' + str(MIN_NUM_INDIVIDUALS_PER_SUBGROUP),
                        'coverage' : 'NA',
                        'no_positive_inst_in_sg':'NA',
                        'no_inst_in_sg' : 'NA'},
                        
                        {'subgroup' : sg_null, 
                        'interestingness' : interestingness_null.item(), 
                        'target_share' : target_share_null.item(),
                        'coverage' : coverage_null.item(),
                        'no_positive_inst_in_sg':no_p_inst_null.item(),
                        'no_inst_in_sg' : no_inst_null.item()}]
    
    prep_for_dataframe = list_to_prepend + prep_for_dataframe

    result_as_df = pd.DataFrame(prep_for_dataframe)

    return result_as_df

def updateBeam(beam, beam_width, unbounded_beam, target_share_null):

    # Taking the top-k subgroup description objects in terms of interestingness, from all the subgroups in the beam
    # And the newly computed ones, but only if the target share is above that of the whole dataset and the 
    # size of the subgroup is above a specified minimum. Also ensures that there are no duplicates because
    # for ex 0=0 AND 0=1 and 0=1 AND 0=0 can both be in the beam
    k = 0

    while len(beam) < beam_width:
        
        try:
            if(unbounded_beam[k].target_share >= target_share_null and 
                        unbounded_beam[k].no_sg_inst > tconst.MIN_NUM_INDIVIDUALS_PER_SUBGROUP and unbounded_beam[k]
                        not in beam):
                beam.append(unbounded_beam[k])

        except IndexError:
            print("Run out of subgroups to try and put in the beam , check if statement conditions")
            break

        k += 1

    return beam


def firstIteration(beam, selectors, data, beam_width):

    unbounded_beam = []

    for sel in selectors:

        sg = beam_search_models_tensor.Conjunction([sel])

        what_sg_covers = sg.covers(data)
        
        interestingness, target_share, coverage, no_sg_p_inst, no_sg_inst = computeMetrics(data, what_sg_covers)
       
        unbounded_beam.append(beam_search_models_tensor.SubgroupDetails(sg, interestingness, target_share, coverage,
                                                                    no_sg_p_inst, no_sg_inst))
    
    # Sort in descending order based on interestingness
    unbounded_beam.sort(key=lambda x: x.interestingness, reverse=True)

    # Take the top-k selectors and keep that in beam
    for k in range(0, beam_width):
        beam.append(unbounded_beam[k])

    return beam

def convertSGtoSet(sg):
    res =set()
    for sel in sg.selectors:
        res.add(sel)
    return res

def add_if_required(result, sg, quality, result_set_size, data, device, MIN_NUM_INDIVIDUALS_PER_SUBGROUP, check_for_duplicates=False):
    if check_for_duplicates and (quality, sg) in result:
        return

    sg_set = convertSGtoSet(sg)
    for pair in result:
        beam_quality = pair[0]
        sg_beam = pair[1]
        sg_beam_set = convertSGtoSet(sg_beam)
        subtract =  sg_beam_set - sg_set
        if len(subtract)==0 and quality < beam_quality:
            # print("Found a subset with better score! Not added")
            return
    no_sg_inst = sg.covers(data, device).sum()
    if no_sg_inst < MIN_NUM_INDIVIDUALS_PER_SUBGROUP:
        return 
    if len(result) < result_set_size:
        heappush(result, (quality, sg))
    elif quality > result[0][0]:
        heappop(result)
        heappush(result, (quality, sg))

def beamSearch(selectors, data, device, max_depth, beam_width, COVERAGE_COEFF, MIN_NUM_INDIVIDUALS_PER_SUBGROUP):
    result_set_size = beam_width
    sg_null, interestingness_null, target_share_null, coverage_null, no_p_inst_null, no_inst_null  = get_info_of_null_sg(data)
    beam = [(0, beam_search_models_tensor.Conjunction([]))]
    last_beam = None
    target = beam_search_models_tensor.EquitySelector(data.shape[1]-1, torch.tensor(1))
    depth = 0
    while beam != last_beam and depth < max_depth:
        last_beam = beam.copy()
        print("last_beam size: {}, depth: {}".format(len(last_beam), depth))
        for (_, last_sg) in last_beam:
            if not getattr(last_sg, 'visited', False):
                setattr(last_sg, 'visited', True)
                for sel in tqdm(selectors):
                    # create a clone
                    new_selectors = list(last_sg.selectors)
                    if sel not in new_selectors:
                        new_selectors.append(sel)
                        sg = beam_search_models_tensor.Conjunction(new_selectors)
                        sg_vector = sg.covers(data, device)
                        outcome_vector = target.covers (data)
                        interestingness, target_share, coverage, no_sg_p_inst, no_sg_inst = computeMetrics(data, sg_vector)
                        quality= interestingness
                        add_if_required(beam, sg, quality, beam_width, data, device, MIN_NUM_INDIVIDUALS_PER_SUBGROUP,  check_for_duplicates=True)
        depth += 1

    result = beam[:result_set_size]
    result.sort(key=lambda x: x[0], reverse=True)
    wrapped_beam = []
    for _, sg in result:
        sg_vector = sg.covers(data, device)
        interestingness, target_share, coverage, no_sg_p_inst, no_sg_inst = computeMetrics(data, sg_vector)
        sgDetail = beam_search_models_tensor.SubgroupDetails(sg, interestingness, target_share, coverage, no_sg_p_inst, no_sg_inst) 
        wrapped_beam.append(sgDetail)
    sgd_df = creatingDataframe(wrapped_beam, sg_null, interestingness_null, target_share_null, coverage_null, no_p_inst_null, no_inst_null, COVERAGE_COEFF, MIN_NUM_INDIVIDUALS_PER_SUBGROUP)
    return sgd_df, wrapped_beam



def get_attributes_specific_selectors(final_beam):

    all_attributes = []

    # For all the subgroup info objects in the final beam we have from beam search
    for sg_obj in final_beam:
        selectors = sg_obj.conjunction.selectors
        attributes_of_this_sg = []

        for sel in selectors:
            if sel.value != 0:
                attributes_of_this_sg.append(sel.attribute)
        
        all_attributes += attributes_of_this_sg

    all_attributes = list(set(all_attributes))

    return all_attributes



def get_attributes_all_selectors(final_beam):

    all_attributes = []

    # For all the subgroup info objects in the final beam we have from beam search
    for sg_obj in final_beam:
        selectors = sg_obj.conjunction.selectors
        attributes_of_this_sg = []

        for sel in selectors:
            attributes_of_this_sg.append(sel.attribute)
        
        all_attributes += attributes_of_this_sg

    all_attributes = list(set(all_attributes))

    return all_attributes




def loss(final_beam, NO_SAMPLES_CONSIDERED_IN_SGD_LOSS):
    desired_average = 0

    # Getting the desired average
    for i in range(0, NO_SAMPLES_CONSIDERED_IN_SGD_LOSS):
        desired_average += final_beam[i].interestingness
        
    desired_average /= NO_SAMPLES_CONSIDERED_IN_SGD_LOSS

    sgd_loss = 1-desired_average 

    return sgd_loss


# Note the binned latents tensor should have the class column at the end
def run(binned_latents_tensor, device, COVERAGE_COEFF, MIN_NUM_INDIVIDUALS_PER_SUBGROUP, NO_SAMPLES_CONSIDERED_IN_SGD_LOSS, MAX_DEPTH, BEAM_WIDTH):

    no_columns = len(binned_latents_tensor[1,:])
    
    # selectors is a list of selectors for every value that every feature has
    selectors = createSelectors(binned_latents_tensor, no_columns)
    sgd_df, final_beam = beamSearch(selectors, binned_latents_tensor, device, MAX_DEPTH, BEAM_WIDTH, COVERAGE_COEFF, MIN_NUM_INDIVIDUALS_PER_SUBGROUP)
    print ("extracted beam:")
    print (final_beam)
    # Shorten the final beam to just the top subgroups for the latents to optimize computation
    final_beam = final_beam[0:NO_SAMPLES_CONSIDERED_IN_SGD_LOSS]

    latents_to_optimize = get_attributes_specific_selectors(final_beam)

    sgd_loss = loss(final_beam, NO_SAMPLES_CONSIDERED_IN_SGD_LOSS)

    return sgd_df, sgd_loss, latents_to_optimize

    

    





    
