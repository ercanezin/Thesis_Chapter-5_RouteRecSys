"""
NOTES:
This python file can run experiment given datasets created for route creation.
"""

import copy
import gc
import sys
from datetime import datetime, timedelta
from math import sqrt
import time


import utils
import threading

import logging
LOG = logging.getLogger(__name__)
log_handler = logging.StreamHandler(sys.stdout)
log_handler.setFormatter(logging.Formatter("%(asctime)s- %(levelname)s: %(message)s "))
log_handler.setLevel(logging.INFO)
logging.FileHandler('info.log')
LOG.addHandler(log_handler)
LOG.setLevel(logging.INFO)

# Defining Global Variables
DISTANCE_MATRIX  = WEATHER_DATA = MEAN_RATING_SCORE_of_BEST_VENUE = MAX_NUMBER_OF_TAG =  CITY_PATH = \
    NUM_OF_ROUTE_TO_PROCESS = MAX_NODE_OWNERSHIP_OF_ROUTE = TOTAL_ROUTE_DURATION_MAX_LIMIT  =  CITY = ROUTES= None

GOOGLE_NODES_DICT=dict()
GOOGLE_NODES_SIMPLE_DICT=dict()
GOOGLE_NODE_ID_LIST=list()
PROB_DIST=list()
MAX_NODE_DEPTH_OF_TREE=0

def calculate_reward(current_node, norm_travel_distance, popular_time_perc, experiment_parameters):
    rating_effect = 0
    if experiment_parameters['rating']:
        rating_factor_weight = 1

        rating, rating_n = utils.get_rating(current_node.node_id,GOOGLE_NODES_DICT)
        global MEAN_RATING_SCORE_of_BEST_VENUE
        rating_effect = rating_factor_weight * (rating / MEAN_RATING_SCORE_of_BEST_VENUE) * (
                sqrt(min(100, rating_n)) / sqrt(100))
    if experiment_parameters['num_of_tags']:
        num_of_tags = len(current_node.google_tags) / 5  # max number of tags for nodes is 5
    else:
        num_of_tags = 0

    if experiment_parameters['travel_distance']:
        near_enough = 1 - norm_travel_distance
    else:
        near_enough = 0

    return rating_effect + popular_time_perc + num_of_tags + near_enough


# common function for original and generated
def route_reward_ranking_calculation(route, route_duration):
    # ranking_quality visit_duration
    ranking = sum([nd.ranking_quality for nd in route]) / (len(route) - 1)
    ranking_time = sum([nd.ranking_quality * nd.visit_duration for nd in route]) / route_duration
    reward = sum([nd.own_reward for nd in route]) / (len(route) - 1)
    reward_time = sum([nd.own_reward * nd.visit_duration for nd in route]) / route_duration

    return ranking, ranking_time, reward, reward_time


# return original route nodes with calculated reward and ranking quality.
def original_route_relative_reward_calculation(user_route_nodes_on_gmap_simple,
                                               user_route_node_ids,
                                               optimum_walking_time_and_distance,
                                               visit_times_list,
                                               experiment_parameters,
                                               weather_on_the_day
                                               ):
    # Let's loop over all nodes except first node till we reach the last node
    for index, current_node in enumerate(user_route_nodes_on_gmap_simple):
        if index + 1 == len(user_route_node_ids):
            continue
        next_node = user_route_nodes_on_gmap_simple[index + 1]

        parent_ids = user_route_node_ids[:index + 1]
        children = utils.get_nodes_in_walking_distance(current_node,
                                                 optimum_walking_time_and_distance,
                                                 parent_ids,
                                                DISTANCE_MATRIX,
                                                GOOGLE_NODES_SIMPLE_DICT,
                                                 set_parent_node=False)

        # We added actual visited child even if it is not in the walking distance to compare the rank--> Ask Seth
        children.append(copy.copy(next_node))

        for child in children:

            weather_weight = 1
            # START: Filter out closed places and outdoor places when weather conditions are severe
            if experiment_parameters['weather']:
                # 1 or 0
                is_weather_outdoor_friendly = utils.get_weather_on_visit_time(weather_on_the_day, visit_times_list[index])
                is_outdoor = True if 'outdoors' in child.google_tags else False

                if is_weather_outdoor_friendly is False and is_outdoor:
                    weather_weight = 0

            is_place_open = 1
            if experiment_parameters['opennes']:
                is_place_open = utils.is_place_open_now(child, visit_times_list[index],GOOGLE_NODES_DICT)  # 1=open,0=close

            if is_place_open == 0 or weather_weight == 0:
                continue

            child.own_reward = calculate_reward_of_transition(current_node,
                                                              child,
                                                              optimum_walking_time_and_distance[1],
                                                              visit_times_list[index],
                                                              experiment_parameters)

        next_node.ranking_quality = utils.calculate_ranking_quality_of_next_node(children)
        next_node.own_reward = children[-1].own_reward
        next_node.visit_duration = utils.get_average_time_spent(next_node.node_id,GOOGLE_NODES_DICT)

    return user_route_nodes_on_gmap_simple




def mcts_process(
        user_route_node_ids,
        weather_on_the_day,
        optimum_walking_time_and_distance,
        file_name,
        experiment_parameters,
        start_time,
        end_time,
        is_legal_dict,
        route_index,
        visit_times_list,
        user_id):

    #create a list of user visited nodes
    stime = datetime.now()
    user_route_nodes_on_gmap_simple = [copy.copy(GOOGLE_NODES_SIMPLE_DICT[node_id]) for node_id in user_route_node_ids]
    global MAX_NODE_DEPTH_OF_TREE
    MAX_NODE_DEPTH_OF_TREE=len(user_route_node_ids)*2

    starting_node = user_route_nodes_on_gmap_simple[0]
    total_route_duration = (end_time - start_time).total_seconds()  # yields timedelta

    starting_node.total_num_of_tree_nodes = 1

    time_spend_constant = 1
    if total_route_duration > TOTAL_ROUTE_DURATION_MAX_LIMIT: # 21600 seconds = 6 hours
        time_spend_constant = 2

    root = expand_nodes(starting_node,
                        total_route_duration,
                        optimum_walking_time_and_distance,
                        [],  # parent_ids
                        start_time,
                        end_time,
                        weather_on_the_day,
                        experiment_parameters,
                        time_spend_constant)

    tree_construction_duration = (datetime.now() - stime).total_seconds()

    # BEST ROUTE EXTRACTION STARTS
    stime = datetime.now()
    best_route = utils.get_best_route_from_tree(root)
    best_route_extraction_duration = (datetime.now() - stime).total_seconds()
    # BEST ROUTE EXTRACTION ENDS

    original_route = original_route_relative_reward_calculation(user_route_nodes_on_gmap_simple,
                                                                user_route_node_ids,
                                                                optimum_walking_time_and_distance,
                                                                visit_times_list,
                                                                experiment_parameters,
                                                                weather_on_the_day)


    if len(best_route) > 2:
        b_ranking, b_ranking_time, b_reward, b_reward_time = route_reward_ranking_calculation(best_route,
                                                                                              total_route_duration)

        o_ranking, o_ranking_time, o_reward, o_reward_time = route_reward_ranking_calculation(original_route,
                                                                                              total_route_duration)

        # SIMILARITY CALCULATION STARTS
        stime = datetime.now()
        best_ndcg, best_jaccard = utils.get_similarity_of_best_route(copy.deepcopy(user_route_nodes_on_gmap_simple),
                                                               copy.deepcopy(best_route))
        similarity_calc_duration = (datetime.now() - stime).total_seconds()
        # SIMILARITY CALCULATION ENDS

        user_route_names = [GOOGLE_NODES_DICT[node.node_id].place_name for node in user_route_nodes_on_gmap_simple]
        best_route_names = [GOOGLE_NODES_DICT[node.node_id].place_name for node in best_route]

        #Write Route Generation and Experiment results to a file


        output_dict= dict()
        output_dict['user_id']= user_id
        output_dict['route_index']= route_index
        output_dict['file_name']= file_name
        output_dict['total_num_of_tree_nodes']= root.total_num_of_tree_nodes
        output_dict['optimum_walking_time']= str(optimum_walking_time_and_distance[0])
        output_dict['is_legal_dict']= is_legal_dict
        output_dict['user_route_names']= user_route_names
        output_dict['best_route_names']= best_route_names
        output_dict['best_ndcg']= best_ndcg
        output_dict['best_jaccard']= best_jaccard
        output_dict['best_route_reward']= b_reward
        output_dict['best_route_reward_time']= b_reward_time
        output_dict['best_route_ranking']= b_ranking
        output_dict['best_route_ranking_time']= b_ranking_time
        output_dict['org_route_reward']= o_reward
        output_dict['org_route_reward_time']= o_reward_time
        output_dict['org_route_ranking']= o_ranking
        output_dict['org_route_ranking_time']= o_ranking_time

        utils.write_output(output_dict,CITY)

        
        LOG.info('  {}- Route Lenght:{}, Best Route Length:{}, Total Time:{} min, NDCG:{}, Jaccard:{},'
                 ' Tree Built Dur:{}, Best Route Built Dur:{}, Similarity Calc Dur:{}'.format(route_index,
                                                                      str(len(user_route_nodes_on_gmap_simple)),
                                                                      str(len(best_route)),
                                                                      total_route_duration/60,
                                                                      best_ndcg,
                                                                      best_jaccard,
                                                                      tree_construction_duration,
                                                                      best_route_extraction_duration,
                                                                      similarity_calc_duration
                                                                      )
                 )
    else:
        LOG.info('  {}- Route Lenght:{}, Best Route Length:{}, Total Time:{} min, Tree Built Dur:{}, Best Route Built Dur:{}'.format(route_index,
                                                                                              str(len(user_route_nodes_on_gmap_simple)),
                                                                                              str(len(best_route)),
                                                                                              total_route_duration / 60,
                                                                                              tree_construction_duration,
                                                                                              best_route_extraction_duration,
                                                                                              )
                 )

def expand_nodes(parent_node,
                 remaining_time,
                 optimum_walking_time_and_distance,
                 parent_ids,
                 start_time,
                 end_time,
                 weather_on_the_day,
                 experiment_parameters,
                 time_spend_constant):
    if remaining_time > 0 and MAX_NODE_DEPTH_OF_TREE >= len(parent_ids):
        parent_node.children = utils.get_nodes_in_walking_distance(parent_node, optimum_walking_time_and_distance,
                                                             parent_ids, DISTANCE_MATRIX,GOOGLE_NODES_SIMPLE_DICT, True)

        if len(parent_node.children) > 0:
            parent_ids.append(parent_node.node_id)

            for i, child in enumerate(parent_node.children):
                # START: Calculate time and distance to spent and travel for a place
                time_spent_at_child = utils.get_average_time_spent(child.node_id,GOOGLE_NODES_DICT)
                travel_distance, travel_duration = utils.get_travel_duration(parent_node, child, DISTANCE_MATRIX)  # meter, seconds
                child_remaining_time = remaining_time - (travel_duration*time_spend_constant)  # in seconds
                child_visiting_time = end_time - timedelta(seconds=child_remaining_time)  # in seconds
                # END:
                weather_weight = 1
                # START: Filter out closed places and outdoor places when weather conditions are severe
                if experiment_parameters['weather']:
                    is_weather_outdoor_friendly = utils.get_weather_on_visit_time(weather_on_the_day,
                                                                            child_visiting_time)  # 1 or 0
                    is_outdoor = True if 'outdoors' in child.google_tags else False

                    if is_weather_outdoor_friendly is False and is_outdoor:
                        weather_weight = 0

                is_place_open = 1
                if experiment_parameters['opennes']:
                    is_place_open = utils.is_place_open_now(child, child_visiting_time,GOOGLE_NODES_DICT)  # 1=open,0=close

                if child_remaining_time < 0 or is_place_open == 0 or weather_weight == 0:
                    parent_node.children.remove(child)
                    continue
                # END:

                expanded_child_node = expand_nodes(copy.copy(child),
                                                   child_remaining_time - (time_spent_at_child * 60),
                                                   optimum_walking_time_and_distance,
                                                   copy.copy(parent_ids),
                                                   start_time,
                                                   end_time,
                                                   weather_on_the_day,
                                                   experiment_parameters,
                                                   time_spend_constant)

                if experiment_parameters['popular_time']:
                    popular_time_perc = utils.get_popular_time_percentage(expanded_child_node.node_id,
                                                                    child_visiting_time,GOOGLE_NODES_DICT) / 100
                else:
                    popular_time_perc = 0

                if experiment_parameters['probability']:
                    global GOOGLE_NODE_ID_LIST
                    global PROB_DIST
                    x = GOOGLE_NODE_ID_LIST.index(parent_node.node_id)
                    y = GOOGLE_NODE_ID_LIST.index(expanded_child_node.node_id)
                    probability_of_visit = PROB_DIST[x][y]
                    expanded_child_node.own_reward += probability_of_visit

                expanded_child_node.own_reward += calculate_reward(expanded_child_node,
                                                                   travel_distance / optimum_walking_time_and_distance[1],
                                                                   popular_time_perc,
                                                                   experiment_parameters)

                expanded_child_node.reward += expanded_child_node.own_reward
                parent_node.reward += expanded_child_node.reward
                parent_node.total_num_of_tree_nodes = int(parent_node.total_num_of_tree_nodes) + int(expanded_child_node.total_num_of_tree_nodes)
                parent_node.children[i] = expanded_child_node

        return parent_node
    else:
        return parent_node

#  Original Route transition reward calculation
def calculate_reward_of_transition(parent_node,
                                   child_node,
                                   optimum_walking_distance,
                                   child_visiting_time,
                                   experiment_parameters):
    travel_distance, travel_duration = utils.get_travel_duration(parent_node, child_node , DISTANCE_MATRIX)

    popular_time_perc = 0
    if experiment_parameters['popular_time']:
        popular_time_perc = utils.get_popular_time_percentage(child_node.node_id, child_visiting_time,GOOGLE_NODES_DICT) / 100

    probability_of_visit = 0
    if experiment_parameters['probability']:
        probability_of_visit = utils.get_probability_of_visit(parent_node.node_id, child_node.node_id,GOOGLE_NODE_ID_LIST,PROB_DIST)

    rating_effect = 0
    if experiment_parameters['rating']:
        rating_factor_weight = 1
        global MEAN_RATING_SCORE_of_BEST_VENUE
        rating, rating_n = utils.get_rating(child_node.node_id,GOOGLE_NODES_DICT)
        rating_effect = rating_factor_weight * (rating / MEAN_RATING_SCORE_of_BEST_VENUE) * (
                sqrt(min(100, rating_n)) / sqrt(100))

    if experiment_parameters['num_of_tags']:
        num_of_tags = len(child_node.google_tags) / 5  # max number of tags for nodes is 5
    else:
        num_of_tags = 0

    if experiment_parameters['travel_distance']:
        near_enough = 1 - (travel_distance / optimum_walking_distance)
        if near_enough < 0:
            near_enough = 0
    else:
        near_enough = 0

    return rating_effect + popular_time_perc + num_of_tags + near_enough + probability_of_visit



def main():
    exp_start_time = datetime.now()

    '''ASSIGN GLOBAL VALUES'''
    global DISTANCE_MATRIX,GOOGLE_NODES_DICT,GOOGLE_NODES_SIMPLE_DICT, GOOGLE_NODE_ID_LIST,\
        WEATHER_DATA,PROB_DIST,MEAN_RATING_SCORE_of_BEST_VENUE, MAX_NUMBER_OF_TAG,CITY_PATH, \
        NUM_OF_ROUTE_TO_PROCESS, MAX_NODE_OWNERSHIP_OF_ROUTE, TOTAL_ROUTE_DURATION_MAX_LIMIT , MAX_NODE_DEPTH_OF_TREE,\
        CITY,ROUTES

    ''' LOAD DATA'''
    CITY='amsterdam'
    #CITY='bristol'
    CITY_PATH='../../datasets/'+CITY+'/'#
    WEATHER_DATA = utils.load_weather_data(CITY_PATH)
    DISTANCE_MATRIX = utils.load_distance_matrix(CITY_PATH)
    GOOGLE_POI_LOCATIONS, GOOGLE_NODE_ID_LIST = utils.load_google_poi_locations(CITY_PATH)
    GOOGLE_NODES_DICT, GOOGLE_NODES_SIMPLE_DICT = utils.create_google_nodes(GOOGLE_POI_LOCATIONS)
    PROB_DIST = utils.load_probability_distribution(CITY_PATH)
    MEAN_RATING_SCORE_of_BEST_VENUE = utils.get_mean_of_best_venue(GOOGLE_NODES_DICT)
    ROUTES = utils.load_routes(CITY_PATH)

    '''SETTINGS START'''
    MAX_NUMBER_OF_TAG = 5 # We counted this for every POI and this is a fixed number
    optimum_walking_distance_list = [[5, 400]
        ,[10, 800],
        [15, 1200],
        [20, 1600],
        [30, 2400],
        [45, 3540]]
    # minutes and meters in distance, human walkin speed approximately 4,8 km/h'''
    #optimum_walking_distance_list = [[5, 400]]  # minutes and meters in distance

    run_experiment_for = ['all_on',
                          'all_off',
                          'probability_only',
                          'probability_off',
                          'weather_off',
                          'weather_only',
                          'rating_only',
                          'rating_off',
                          'no_of_tags_only',
                          'no_of_tags_off',
                          'popular_time_only',
                          'popular_time_off',
                          'travel_distance_off',
                          'travel_distance_only',
                          'opennes_only',
                          'opennes_off']
    run_experiment_for = ['all_on',
                          'all_off',
                          'probability_only',
                          'weather_only',
                          'rating_only',
                          'no_of_tags_only',
                          'popular_time_only',
                          'travel_distance_only',
                          'opennes_only']

    # probability, rating, popular time, weather, open/close,travel_distance, num of tags

    exp_parameters = {
        'all_on': {'probability': True, 'rating': True, 'popular_time': True, 'weather': True, 'opennes': True,
                   'travel_distance': True, 'num_of_tags': True},

        'all_off': {'probability': False, 'rating': False, 'popular_time': False, 'weather': False,
                    'opennes': False, 'travel_distance': False, 'num_of_tags': False},

        'probability_only': {'probability': True, 'rating': False, 'popular_time': False, 'weather': False,
                             'opennes': False, 'travel_distance': False, 'num_of_tags': False},

        'probability_off': {'probability': False, 'rating': True, 'popular_time': True,

                            'weather': True, 'opennes': True, 'travel_distance': True,
                            'num_of_tags': True},
        'weather_off': {'probability': True, 'rating': True, 'popular_time': True, 'weather': False,
                        'opennes': True, 'travel_distance': True, 'num_of_tags': True},

        'weather_only': {'probability': False, 'rating': False, 'popular_time': False, 'weather': True,
                         'opennes': False, 'travel_distance': False, 'num_of_tags': False},

        'rating_only': {'probability': False, 'rating': True, 'popular_time': False, 'weather': False,
                        'opennes': False, 'travel_distance': False, 'num_of_tags': False},

        'rating_off': {'probability': True, 'rating': False, 'popular_time': True, 'weather': True,
                       'opennes': True, 'travel_distance': True, 'num_of_tags': True},

        'no_of_tags_only': {'probability': False, 'rating': False, 'popular_time': False, 'weather': False,
                            'opennes': False, 'travel_distance': False, 'num_of_tags': True},

        'no_of_tags_off': {'probability': True, 'rating': True, 'popular_time': True, 'weather': True,
                           'opennes': True, 'travel_distance': True, 'num_of_tags': False},

        'popular_time_only': {'probability': False, 'rating': False, 'popular_time': True, 'weather': False,
                              'opennes': False, 'travel_distance': False, 'num_of_tags': False},

        'popular_time_off': {'probability': True, 'rating': True, 'popular_time': False,
                             'weather': True, 'opennes': True, 'travel_distance': True, 'num_of_tags': True},

        'travel_distance_off': {'probability': True, 'rating': True, 'popular_time': True, 'weather': True,
                                'opennes': True, 'travel_distance': False, 'num_of_tags': True},

        'travel_distance_only': {'probability': False, 'rating': False, 'popular_time': False,
                                 'weather': False, 'opennes': False, 'travel_distance': True,
                                 'num_of_tags': False},

        'opennes_only': {'probability': False, 'rating': False, 'popular_time': False, 'weather': False,
                         'opennes': True, 'travel_distance': False, 'num_of_tags': False},

        'opennes_off': {'probability': True, 'rating': True, 'popular_time': True, 'weather': True,
                        'opennes': False, 'travel_distance': True, 'num_of_tags': True}
    }

    NUM_OF_ROUTE_TO_PROCESS=-1
    MAX_NODE_OWNERSHIP_OF_ROUTE=20


    # if total route duration is more than 21600 we multiply transition time by 2 check expand_nodes function for more info
    TOTAL_ROUTE_DURATION_MAX_LIMIT=21600
    '''SETTINGS END'''

    LOG.info("OPTIMUM WALKING LIST:{}".format([owd[0] for owd in optimum_walking_distance_list]))
    LOG.info("EXPERIMENT SETTINGS:{}".format(str(run_experiment_for)))

    for optimum_walking_time_and_distance in optimum_walking_distance_list:
        LOG.info("Total # of Experiments :{}".format(len(run_experiment_for) * len(optimum_walking_distance_list)))
        LOG.info("#############EXPERIMENTS STARTED###########")

        for file_name in run_experiment_for:
            loc_stime = datetime.now()
            LOG.info('->STARTING WITH EXPERIMENT: {0} - OWD: {1}'.format(file_name.replace('_', ' '),
                                                                         str(optimum_walking_time_and_distance[0])))
            threads = []
            for route_index, route in enumerate(ROUTES[:NUM_OF_ROUTE_TO_PROCESS]):
                # empty memory before proceeding next
                gc.collect()
                route_start_time = datetime.strptime(route['start_time'], '%Y-%m-%d %H:%M:%S')
                route_end_time = datetime.strptime(route['end_time'], '%Y-%m-%d %H:%M:%S')
                visit_times_list = [datetime.strptime(visited_at_str, '%Y-%m-%d %H:%M:%S') for visited_at_str in
                                    route['visit_times_list']]
                total_route_duration = (route_end_time - route_start_time).total_seconds()
                if len(route['gmap_node_ids']) > MAX_NODE_OWNERSHIP_OF_ROUTE or total_route_duration > TOTAL_ROUTE_DURATION_MAX_LIMIT:
                    LOG.info('  [{}]-SKIPPED--> No of Node: {}, Route Duration:{} '.format(route_index + 1,len(route['gmap_node_ids']),total_route_duration ))
                    continue

                weather_on_the_day = None
                if exp_parameters[file_name]['weather']:
                    weather_on_the_day = WEATHER_DATA[
                        datetime.strptime(route['start_time'], '%Y-%m-%d %H:%M:%S').strftime('%Y-%m-%d')]

                LOG.info('  [{}]-Started Processing'.format(route_index + 1))


                thread = threading.Thread(target=mcts_process, args=(
                    route['gmap_node_ids'],
                    weather_on_the_day,
                    optimum_walking_time_and_distance,
                    file_name,
                    exp_parameters[file_name],
                    route_start_time,
                    route_end_time,
                    route['is_legal_dict'],
                    route_index + 1,
                    visit_times_list,
                    route['user_id']))

                threads.append(thread)
                thread.start()
                time.sleep(1)


                '''mcts_process(
                    route['gmap_node_ids'],
                    weather_on_the_day,
                    optimum_walking_time_and_distance,
                    file_name,
                    exp_parameters[file_name],
                    route_start_time,
                    route_end_time,
                    route['is_legal_dict'],
                    route_index + 1,
                    visit_times_list,
                    route['user_id']
                )'''

                LOG.info('  [{}]-Ended Processing'.format(route_index + 1))

            for thread in threads:
                thread.join()

            LOG.info('->ENDED- PARAMETERS: {0} - OWD: {1} - DUR: {2} mins'.format(
                file_name.replace('_', ' ').upper(),
                str(optimum_walking_time_and_distance[0]),
                (datetime.now() - loc_stime).total_seconds() / 60)
            )
            LOG.info("#################################################################")

        LOG.info("STATS: run time in secs:{}".format((datetime.now() - exp_start_time).total_seconds()))
        LOG.info("STATS: run time in min:{}".format((datetime.now() - exp_start_time).total_seconds() / 60))
        LOG.info("#############EXPERIMENTS ENDED ###########")

if __name__ == '__main__':
    main()
