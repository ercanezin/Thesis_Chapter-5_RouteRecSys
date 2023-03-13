import numpy as np


class Node:

    def __init__(self, node_id, place_name, coordinates, address, types, google_tags, current_popularity, populartimes,
                 opening_hours, rating, rating_n, time_spent, visited_at=None, parent_id=None, weather_when_visited=None,
                 popular_time_perc=None, time_spent_here=None):
        self.node_id = node_id
        self.parent_id = parent_id
        self.total_num_of_tree_nodes = 0
        self.children = list()
        self.place_name = place_name
        self.coordinates = coordinates
        self.popular_times = populartimes
        self.address = address
        self.types = types
        self.google_tags = google_tags
        self.current_popularity = current_popularity
        self.opening_hours = opening_hours
        self.rating = rating
        self.rating_n = rating_n
        self.time_spent = time_spent
        self.reward = 0.
        self.own_reward = 0.
        self.visited_at = visited_at
        self.weather_when_visited = weather_when_visited
        self.popular_time_perc = popular_time_perc
        self.time_spent_here = time_spent_here

    def __repr__(self):
        return "Node id:{0}  place_name:{1} coordinates :{2}  address:{3}   googletags:{4}  types:{5}  populartimes:{" \
               "6}  current_popularity:{7}  children:{8}  opening_hours:{9}".format(
            self.node_id,
            self.place_name,
            self.coordinates,
            self.address,
            self.google_tags,
            self.types,
            self.popular_times,
            self.current_popularity,
            self.children,
            self.opening_hours,
            self.rating,
            self.rating_n,
            self.time_spent,
            self.visited_at
        )

    def best_child(self):
        return self.children[np.argmax([child.reward for child in self.children])]