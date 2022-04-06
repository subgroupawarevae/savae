import pandas as pd
import numpy as np
from modules import beam_search_models_tensor 
from modules import training_constants as tconst
import torch


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


def creatingDataframe(beam, sg_null, interestingness_null, target_share_null, coverage_null, no_p_inst_null, no_inst_null):

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
                        'interestingness': 'COVERAGE_COEFF: ' + str(tconst.COVERAGE_COEFF), 
                        'target_share' : 'MIN NO INDIVIDUALS PER SG: ' + str(tconst.MIN_NUM_INDIVIDUALS_PER_SUBGROUP),
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

        

def beamSearch(selectors, data, max_depth, beam_width):

    # Compute statistics for null subgroup 
    sg_null, interestingness_null, target_share_null, coverage_null, no_p_inst_null, no_inst_null  = get_info_of_null_sg(data)
    tuple_to_show = get_info_of_null_sg(data)

    beam = []
    last_beam = beam.copy()
    evaluated_depth = 0

    # If the desired beam width is greater then the no of len 1 selectors we have to just fill with the 
    # max no of selectors we have
    if(beam_width > tconst.NO_LEN_1_SELECTORS):
        # Now beam is full of the top k subgroup description objects in terms of interestingness
        beam = firstIteration(beam, selectors, data, tconst.NO_LEN_1_SELECTORS)
    else:
        beam = firstIteration(beam, selectors, data, beam_width)

    print ("after first iteratin:")
    print (beam)
    # Just did the first iteration
    evaluated_depth += 1


    i = 0

    # While the beam changed last iteration, and we should still be increasing the depth of our lists of selectors
    while beam != last_beam and evaluated_depth < max_depth:

        # So we know if beam has changed
        last_beam = beam.copy()

        newly_generated_sgs_list = []
        unbounded_beam = []

        # For each subgroup in the beam
        for sg_and_details_object in beam:
            
            # If the subgroup can be expanded
            if(len(sg_and_details_object.conjunction) < max_depth):
                
                # Tack on a new selector and compute the metrics of this new subgroup
                for sel in selectors:
                    
                    # No need to add a selector that is already there
                    if (sel not in sg_and_details_object.conjunction.selectors):
                        
                        # Copy list of selectors 
                        new_selectors = sg_and_details_object.conjunction.selectors.copy()

                        # Append the new selector
                        new_selectors.append(sel)

                        # Make list into a Conjunction therefore creating an sg
                        new_sg = beam_search_models_tensor.Conjunction(new_selectors)

                        # print(new_sg)

                        # Compute metrics and append 
                        what_sg_covers = new_sg.covers(data)

                        # print(what_sg_covers)

                        interestingness, target_share, coverage, no_sg_p_inst, no_sg_inst = computeMetrics(data, what_sg_covers)

                        newly_generated_sgs_list.append(beam_search_models_tensor.SubgroupDetails(new_sg, interestingness, 
                                                                        target_share, coverage, no_sg_p_inst, no_sg_inst))

                        # if i == 4:
                        #     print(data)
                        #     return

                        # i += 1


        i += 1

        unbounded_beam = beam + newly_generated_sgs_list
        
        unbounded_beam.sort(key=lambda x: x.interestingness, reverse=True)

        beam.clear()

        beam = updateBeam(beam, beam_width, unbounded_beam, target_share_null)

        evaluated_depth += 1

    sgd_df = creatingDataframe(beam, sg_null, interestingness_null, target_share_null, coverage_null, no_p_inst_null, no_inst_null)
    
    return sgd_df, beam


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




def loss(final_beam):
    print ("fianl_beam:")
    print (final_beam)
    desired_average = 0

    # Getting the desired average
    for i in range(0, tconst.NO_SAMPLES_CONSIDERED_IN_SGD_LOSS):
        desired_average += final_beam[i].interestingness
        
    desired_average /= tconst.NO_SAMPLES_CONSIDERED_IN_SGD_LOSS

    sgd_loss = 1-desired_average 

    return sgd_loss


# Note the binned latents tensor should have the class column at the end
def run(binned_latents_tensor):

    no_columns = len(binned_latents_tensor[1,:])
    
    # selectors is a list of selectors for every value that every feature has
    selectors = createSelectors(binned_latents_tensor, no_columns)
    print (selectors)
    print (binned_latents_tensor)
    sgd_df, final_beam = beamSearch(selectors, binned_latents_tensor, tconst.MAX_DEPTH, tconst.BEAM_WIDTH)
    print (final_beam)
    # Shorten the final beam to just the top subgroups for the latents to optimize computation
    final_beam = final_beam[0:1]

    latents_to_optimize = get_attributes_specific_selectors(final_beam)

    sgd_loss = loss(final_beam)

    return sgd_df, sgd_loss, latents_to_optimize

    

    





    
