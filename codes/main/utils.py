"""
NOTES:
This python file can run experiment given datasets created for route creation.


"""

import ast
import copy
import math
import os
import sys
import time
from datetime import datetime, timedelta
from math import sqrt
 
from Node import Node
from SimpleNode import SimpleNode

import logging
LOG = logging.getLogger(__name__)
log_handler = logging.StreamHandler(sys.stdout)
log_handler.setFormatter(logging.Formatter("%(asctime)s- %(levelname)s: %(message)s "))
log_handler.setLevel(logging.INFO)
logging.FileHandler('info.log')
LOG.addHandler(log_handler)
LOG.setLevel(logging.INFO)

def load_weather_data(city_path):
    start_time = datetime.now()
    LOG.info('Loading WEATHER started')
    # Time, Temperature, Dew, Point, Humidity, Wind, Wind, Speed, Wind, Gust, Pressure, Precip., Condition
    with open(city_path+'weather_complete', encoding="utf8") as f:
        weather = f.readlines()

    master_weather_dict = dict()
    for item in weather:
        master_weather_dict.update(ast.literal_eval(item))
    LOG.info('Loading WEATHER ended in {} seconds'.format((datetime.now() - start_time).total_seconds()))
    return master_weather_dict

def load_google_poi_locations(city_path):
    start_time = datetime.now()
    LOG.info('Loading GOOGLE LOCATIONS started')
    with open(city_path+'google_pois', encoding="utf8") as f:
        ams_combined = f.readlines()
    google_poi_list = [ast.literal_eval(item) for item in ams_combined]
    google_poi_id_list= [item['id'] for item in google_poi_list]
    LOG.info('Loading GOOGLE LOCATIONS ended in {} seconds'.format((datetime.now() - start_time).total_seconds()))
    return google_poi_list, google_poi_id_list

def load_routes(city_path):
    start_time = datetime.now()
    LOG.info('Loading CONVERTED ROUTES')

    with open(city_path+'routes_google_nodes_converted_with_visit_times', encoding="utf8") as f:
        ams_nodes_converted = f.readlines()
    bc = [ast.literal_eval(item) for item in ams_nodes_converted]
    LOG.info('Loading CONVERTED ROUTES ended in {} seconds'.format((datetime.now() - start_time).total_seconds()))
    return bc


def is_place_open_now(child, child_visiting_time,google_nodes_dict):
    child = google_nodes_dict[child.node_id]
    if child.opening_hours is None:
        # getting average based on tags
        open_count = 0
        close_count = 0
        for node_id,node in google_nodes_dict.items():
            if node.opening_hours is not None and (
                    all(elem in node.google_tags for elem in child.google_tags)
                    or
                    all(elem in node.types for elem in child.types)):
                if is_place_open_now(node, child_visiting_time,google_nodes_dict) == 1:
                    open_count += 1
                else:
                    close_count += 1

        if open_count >= close_count:
            return 1
        else:
            return 0

    elif len(child.opening_hours['periods']) == 1:  # open for 24 hours
        return 1
    elif len(child.opening_hours['periods']) < child_visiting_time.weekday() + 1:
        return 0
    # opening time and closing time is coded as 1700, so we multiply hour by 1000 to get the same format.
    child_visit_time_int = child_visiting_time.hour * 100 + child_visiting_time.minute
    opens_at = int(child.opening_hours['periods'][child_visiting_time.weekday()]['open']['time'])
    closes_at = int(child.opening_hours['periods'][child_visiting_time.weekday()]['close']['time'])
    if opens_at <= child_visit_time_int <= closes_at:
        return 1
    else:
        return 0


def get_popular_time_percentage(expanded_node_id, child_visiting_time, google_nodes_dict ):
    expanded_child = google_nodes_dict[expanded_node_id]
    if child_visiting_time.minute > 30:
        hour = (child_visiting_time.hour + 1) % 24
    else:
        hour = child_visiting_time.hour

    if expanded_child.popular_times:
        return expanded_child.popular_times[child_visiting_time.weekday()]['data'][hour]

    else:
        # get tag average
        cnt = 0
        sum_total = 0
        for k,node in google_nodes_dict.items():

            if node.popular_times is not None and (
                    all(elem in node.google_tags for elem in expanded_child.google_tags) or all(elem in node.types for elem in expanded_child.types)):
                sum_total += get_popular_time_percentage(node.node_id, child_visiting_time,google_nodes_dict)
                cnt += 1

        if cnt != 0:
            return int(sum_total / cnt)
        else:
            return 50




def is_outdoor_friendly(condition):
    conditions = {'Partial Fog': True, 'Fair': True, 'Light Rain Shower': False, 'Heavy Rain / Windy': False,
                  'Patches of Fog': True,
                  'Showers in the Vicinity': False, 'Partly Cloudy / Windy': True, 'Mist': True,
                  'Light Drizzle / Windy': True,
                  'Fair / Windy': True, 'Light Rain / Windy': False, 'Cloudy / Windy': True, 'Fog': True,
                  'Mostly Cloudy / Windy': True,
                  'Rain Shower': False, 'Light Drizzle': True, 'Rain': False, 'Heavy Rain': False,
                  'Thunder in the Vicinity': False,
                  'Haze': True,
                  'Rain and Sleet': False,
                  'Light Rain': False, 'Shallow Fog': True, 'Fog / Windy': True, 'Light Rain Shower / Windy': False,
                  'Rain / Windy': False,
                  'Wintry Mix': False, 'Partly Cloudy': True, 'Mostly Cloudy': True, 'Rain Shower / Windy': False,
                  'Light Snow': False,
                  'Heavy Snow': False,
                  'Snow': False,
                  'T-Storm': False, 'Drizzle': False, 'Cloudy': True}
    return conditions[str.strip(condition)]

def get_weather_on_visit_time(weather_on_the_day, child_visit_time):
    begin_time_minute = datetime.strptime(weather_on_the_day[0][0], "%I:%M %p").minute
    next_time_minute = datetime.strptime(weather_on_the_day[1][0], "%I:%M %p").minute

    # We consider given value on an x-axis with 180 values
    # 0.......a=[50]...60...b=[50+begin_time]...c=[60+next_time_minute]...120...d=[120+begin_time]
    child_visit_time_minute = child_visit_time.minute + 60
    a = next_time_minute
    b = 60 + begin_time_minute
    c = 60 + a
    d = 120 + begin_time_minute

    closest = min([a, b, c, d], key=lambda x: abs(x - child_visit_time_minute))

    visit_time_value = None
    if closest == a:
        visit_time_value = (child_visit_time.replace(minute=0, second=0) + timedelta(
            minutes=next_time_minute, hours=-1)).strftime("%I:%M %p")
    elif closest == b:
        visit_time_value = (child_visit_time.replace(minute=0, second=0) + timedelta(
            minutes=begin_time_minute)).strftime("%I:%M %p")
    elif closest == c:
        visit_time_value = (child_visit_time.replace(minute=0, second=0) + timedelta(
            minutes=next_time_minute)).strftime("%I:%M %p")
    elif closest == d:
        visit_time_value = (child_visit_time.replace(minute=0, second=0) + timedelta(
            minutes=begin_time_minute, hours=1)).strftime("%I:%M %p")

    if visit_time_value[0] == '0':
        visit_time_value = visit_time_value.replace('0', '', 1)

    for hr_weather in weather_on_the_day:
        if str.strip(hr_weather[0]) == visit_time_value:
            return is_outdoor_friendly(hr_weather[-1])

def get_rating(c_node_id,google_nodes_dict):
    current_node = google_nodes_dict[c_node_id]
    'returns in minutes'
    if current_node.rating and current_node.rating_n:
        return current_node.rating, current_node.rating_n
    else:
        # get_average
        cnt = 0
        sum_rat = 0
        sum_rat_n = 0
        for node_id, node in google_nodes_dict.items():
            if node.rating is not None and node.rating_n is not None and (
                    all(elem in node.google_tags for elem in current_node.google_tags)
                    or
                    all(elem in node.types for elem in current_node.types)):
                sum_rat += node.rating
                sum_rat_n += node.rating_n
                cnt += 1
        if cnt != 0:
            return sum_rat / cnt, int(sum_rat_n / cnt)
        else:
            return 2.5, 100  # just an average



def get_average_time_spent(node_id,google_nodes_dict):
    """Returns an average integer"""
    ch = google_nodes_dict[node_id]
    if ch.time_spent is not None:
        return (ch.time_spent[0] + ch.time_spent[1]) / 2
    else:
        cnt = 0
        max_v = 0
        min_v = 0
        for n_id, node in google_nodes_dict.items():
            if node.time_spent is not None and (
                    all(elem in node.google_tags for elem in node.google_tags) or
                    all(elem in node.types for elem in node.types)):
                min_v += node.time_spent[0]
                max_v += node.time_spent[1]
                cnt += 1

        if cnt != 0:
            return (max_v + min_v) / 2
        else:
            return 30  # just an average time_spent in minutes

def load_distance_matrix(city_path):
    start_time = datetime.now()
    LOG.info('Loading DISTANCE_MATRIX started')
    with open(city_path+'distance_matrix') as f:
        dm = f.readlines()
    matrix = [ast.literal_eval(item) for item in dm]
    LOG.info('Loading DISTANCE_MATRIX ended in {} seconds'.format((datetime.now() - start_time).total_seconds()))
    return matrix[0]

def create_google_nodes(google_locations):
    start_time = datetime.now()
    LOG.info('Loading GOOGLE NODES started')
    google_nodes = []
    google_nodes_simple = []
    for loc in google_locations:
        g_id = loc['id']
        loc_name = loc['name']
        coordinates = loc['coordinates']
        address = None
        types = loc['types'] if 'types' in loc else []
        google_tags = loc['google_tags'] if 'google_tags' in loc else []
        current_popularity = loc['current_popularity'] if 'current_popularity' in loc else None
        populartimes = loc['populartimes'] if 'populartimes' in loc else None
        opening_hours = loc['opening_hours'] if 'opening_hours' in loc else None
        rating = loc['rating'] if 'rating' in loc else None
        rating_n = loc['rating_n'] if 'rating_n' in loc else None
        time_spent = loc['time_spent'] if 'time_spent' in loc else None

        node = Node(g_id, loc_name, coordinates, address, types, google_tags, current_popularity, populartimes,
                        opening_hours, rating, rating_n, time_spent)

        node_simple = SimpleNode(g_id, opening_hours, types, google_tags)

        google_nodes.append(node)
        google_nodes_simple.append(node_simple)


    google_nodes_simple_dict =  {node_simple.node_id: node_simple for node_simple in google_nodes_simple}
    google_nodes_dict = {node.node_id: node for node in google_nodes}
    LOG.info('Loading GOOGLE NODES ended in {} seconds'.format((datetime.now() - start_time).total_seconds()))
    return google_nodes_dict, google_nodes_simple_dict

def get_simple_node_by_id(node_id,google_nodes_simple_dict):
    return google_nodes_simple_dict[node_id]

def get_best_route_from_tree(root):
    the_best = None
    if len(root.children) > 0:
        the_best = get_best_route_from_tree(root.best_child())

    if the_best is None:
        return [root]
    else:
        the_best.insert(0, root)
        return the_best

def find_most_similar_route(all_routes, actual_route):
    actual_route_list = [poi.node_id for poi in actual_route]
    actual_route_list.pop(0)
    normal_ndcg = 0
    for i, node in enumerate(actual_route_list):
        normal_ndcg += (len(actual_route_list) - i) / (math.log2(i + 2))
        # Relevance/Log(order of item +1) -> We choose relevance as the order of item or we can
        # TODO: alternatively choose 1 or 0 later and try 2^relevance

    similarity_normalised_ndcg_dict = dict()
    similarity_jaccard = list()
    s_actual = set(actual_route_list)
    for route in all_routes:
        my_route = route[1:]
        route_conversion = [(actual_route_list.index(nd.node_id) + 1) / (
            math.log2(my_route.index(nd) + 2)) if nd.node_id in actual_route_list else 0 for nd in my_route]

        similarity_normalised_ndcg_dict[all_routes.index(route)] = sum(route_conversion) / normal_ndcg

        s_route = set(poi.node_id for poi in my_route)
        similarity_jaccard.append((len(s_route.intersection(s_actual)) / len(s_route.union(s_actual))))

    max_index = max(similarity_normalised_ndcg_dict, key=similarity_normalised_ndcg_dict.get)

    return all_routes[max_index], similarity_normalised_ndcg_dict[max_index], normal_ndcg, similarity_jaccard[
        max_index], 1

def get_similarity_of_best_route(actual_route, best_route):
    actual_route_list = [poi.node_id for poi in actual_route[1:]]
    best_route_list = [poi.node_id for poi in best_route[1:]]

    normal_ndcg = 0
    for i, node in enumerate(actual_route_list):
        normal_ndcg += (len(actual_route_list) - i) / (math.log2(i + 2))

    route_conversion = [(actual_route_list.index(nd_id) + 1) / (
        math.log2(best_route_list.index(nd_id) + 2)) if nd_id in actual_route_list else 0 for nd_id in best_route_list]

    actual_route_set = set(actual_route_list)
    best_route_set = set(best_route_list)

    similarity_jaccard = (len(actual_route_set.intersection(best_route_set)) /
                          len(actual_route_set.union(best_route_set))
                          )

    # return ndcg and jaccard
    return (sum(route_conversion) / normal_ndcg), similarity_jaccard

def calculate_ranking_quality_of_each_children(children):
    sorted_children = sorted(children, key=lambda nd: nd.own_reward)
    for ind, child in enumerate(sorted_children):
        child.ranking_quality = (ind + 1) / len(children)
    return sorted_children

# To find the ranking quality of actual visited node
def calculate_ranking_quality_of_next_node(children):
    next_node = children[-1]
    sorted_children = sorted(children, key=lambda child: child.own_reward)
    return (sorted_children.index(next_node) + 1) / len(sorted_children)

def get_travel_duration(node_1, node_2, distance_matrix):
    return distance_matrix[node_1.node_id][node_2.node_id]  # returns a tuple in distance(meter),duration(second)

def get_probability_of_visit(parent_node_id, next_node_id,google_node_id_list,prob_dist):
    x = google_node_id_list.index(parent_node_id)
    y = google_node_id_list.index(next_node_id)
    return prob_dist[x][y]

def get_nodes_in_walking_distance(root_node, optimum_walking_time_and_distance, parent_ids, distance_matrix, google_nodes_simple_dict, set_parent_node=False):
    proximal_nodes = []
    dist_dict = distance_matrix[root_node.node_id]

    for proximal_node_id, dist_dur_tuple in dist_dict.items():

        if dist_dur_tuple[1] <= optimum_walking_time_and_distance[1]:  # (dist,duraction)
            if root_node.parent_id is not None:
                dist_to_parent = distance_matrix[root_node.parent_id][proximal_node_id]
                if dist_to_parent[1] < optimum_walking_time_and_distance[1]:
                    continue

            if proximal_node_id in parent_ids or proximal_node_id == root_node.node_id:
                continue

            proximal_node =  google_nodes_simple_dict[proximal_node_id]

            pr_node_cp = copy.copy(proximal_node)
            if set_parent_node:
                pr_node_cp.parent_id = root_node.node_id
                pr_node_cp.total_num_of_tree_nodes = 1
            proximal_nodes.append(pr_node_cp)
        else:
            continue

    return proximal_nodes

def get_mean_of_best_venue(google_nodes):
    start_time = datetime.now()
    LOG.info('Getting MEAN_OF_BEST_VENUE')
    best = 0
    for nd_id,nd in google_nodes.items():
        if nd.rating and nd.rating > best:
            best = nd.rating
    LOG.info('Getting MEAN_OF_BEST_VENUE ended in {} seconds'.format((datetime.now() - start_time).total_seconds()))
    return best

def load_probability_distribution(city_path):
    start_time = datetime.now()
    LOG.info('Loading PROB_DISTRIBUTION started')
    with open(city_path+'prob_distributions') as f:
        dm = f.readlines()
    matrix = [ast.literal_eval(item) for item in dm]
    LOG.info('Loading PROB_DISTRIBUTION ended in {} seconds'.format((datetime.now() - start_time).total_seconds()))
    return matrix[0]

def write_output(output_dict,city): 
    with open('../../evaluation/'+city+'/'+output_dict['file_name']+'_'+output_dict['optimum_walking_time'], 'a+',encoding='utf-8') as myfile:
        myfile.writelines(str(output_dict) + '\n') 

 