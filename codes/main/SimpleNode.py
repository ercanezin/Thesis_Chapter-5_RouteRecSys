import numpy as np


class SimpleNode:

    def __init__(self, node_id, opening_hours, types, google_tags):
        self.node_id = node_id
        self.parent_id = None
        self.types = types
        self.google_tags = google_tags
        self.num_of_visits = 0
        self.children = list()
        self.opening_hours = opening_hours
        self.reward = 0.
        self.own_reward = 0.
        self.ranking_quality = 0.
        self.visit_duration = 0.
        self.visited_at = None

    def __repr__(self):
        return "Node id:{0}   children:{1}  opening_hours:{2}  google_tags:{3} types:{4}".format(
            self.node_id,
            self.children,
            self.opening_hours,
            self.google_tags,
            self.types
        )

    def best_child(self):
        return self.children[np.argmax([child.reward for child in self.children])]
