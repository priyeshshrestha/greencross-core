# -*- coding: utf-8 -*-
"""
PeopleTracker class 
Author: Avanish Shrestha, 2018
"""

from __future__ import division, print_function, absolute_import

import os
import sys
import logging
import datetime
import numpy as np
import cv2
from app.helper.person import Person
from smartvision.util.coord import get_midpoint, get_distance

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

def calc_threshold(y, tilt, min_t, max_t):
    threshold = min_t + pow(y, (1/tilt))
    threshold = min_t if threshold < min_t else threshold
    threshold = max_t if threshold > max_t else threshold
    return int(threshold)

class PeopleTracker(object):
    """PeopleTracker class.
    
    Attributes:
        people: List of Person objects
        trackers: List of coorindates
        pass_boundaries: List of Passerby Boundary objects
        entry_boundaries: List of Entry/Exit Boundary objects
        mongodb: MongoDB object
        entries: Integer value of current entries
        exits: Integer value of current exits
        passed: Integer value of current passerbys
        count: Integer value of current count for no. of people in the frame
    """

    def __init__(self, pass_boundaries, entry_boundaries, entry_polygons, mongodb, entries=0, exits=0, passed=0, count=0,
                 tracker_id=0, data=None):
        """Object Initializer."""
        self.uuids = []
        self.people = []
        self.coordinates = []
        self.pass_boundaries = pass_boundaries
        self.entry_boundaries = entry_boundaries
        self.entry_polygons = entry_polygons
        self.mongodb = mongodb
        self.entries = entries
        self.exits = exits
        self.passed = passed
        self.count = count
        self.people_id = tracker_id
        self.data = data

    def update(self, points=[], update_type="distance"):
        """Associate the existing people trackers with new detected points.

        Args:
            points: List of detected points
        """
        self.coordinates = []
        self.uuids = []
        self.count = len(points)
        people_ids = []
        new_people = []
        for point in points:
            scores = []
            for person in self.people:
                if person.id in people_ids:
                    scores.append(1000)
                else:
                    pred_mid = get_midpoint(person.tracker.predict())
                    det_mid = get_midpoint(point)
                    scores.append(get_distance(pred_mid, det_mid))
            closest_point = np.amin(scores) if scores else 1000
            y_coord = get_midpoint(point)[1]
            distance_threshold = calc_threshold(y_coord, self.data["tilt"], self.data["minThreshold"], self.data["maxThreshold"])
            if closest_point <= distance_threshold:
                idx = scores.index(closest_point)
                person = self.people[idx]
                updated_coords = person.update(point)
                self.people[idx].add_coordinates(updated_coords)
                self.coordinates.append(updated_coords)
                self.uuids.append(self.people[idx].id)
                people_ids.append(self.people[idx].id)
            else:
                person = Person(self.people_id, len(self.pass_boundaries), len(self.entry_boundaries), len(self.entry_polygons))
                updated_coords = person.update(point)
                person.add_coordinates(point)
                self.coordinates.append(point)
                self.uuids.append(person.id)
                self.people_id += 1
                new_people.append(person)
        for person in self.people:
            if person.id not in people_ids:
                pred = person.predict(person.coordinates[-1])
                person.add_coordinates(pred)
                self.coordinates.append(pred)
                self.uuids.append(person.id)
        self.people.extend(new_people)

    def check(self):
        """Function to track entry/exit/passby of every person. """
        to_del = []
        for person in self.people:
            if person.predictions == 12:
                to_del.append(person)
            else:
                person.check_pass(self.pass_boundaries)
                person.check_entry(self.entry_boundaries)
                person.check_entry_polygon(self.entry_polygons)
                record = {
                    "tracker_id": person.id,
                }
                if person.did_enter:
                    self.entries += 1
                    record["type"] = "in"
                    self.mongodb.save_count_detail(record)
                    person.did_enter = False
                    logging.info("New Entry. Count: %s" % self.entries)
                if person.did_exit:
                    self.exits += 1
                    record["type"] = "out"
                    self.mongodb.save_count_detail(record)
                    person.did_exit = False
                    logging.info("New Exit. Count: %s" % self.exits)
                if person.did_pass:
                    self.passed += 1
                    record["type"] = "pass"
                    self.mongodb.save_count_detail(record)
                    person.did_pass = False
        for toX in to_del:           
            self.people.remove(toX)

    def get_coordinates(self):
        """Get number of people tracking coordinates."""
        return self.coordinates, self.uuids

    def get_coordinates_dictionary(self):
        """Get the coorinates in dictionary format."""
        data_dict = {}
        for person in self.people:
            coord = person.coordinates[-1]
            center = get_midpoint(coord)
            data_dict[person.id] = {
                "coord": coord,
                "center": center,
            }
        return data_dict

    def get_data(self, type="array"):
        """Returns entries & exits data."""
        if type == "array":
            result = [self.entries, self.exits, self.passed, self.count, str(datetime.datetime.now())]
            return result
        elif type == "dict":
            result = {}
            result["in"] = self.entries
            result["out"] = self.exits
            result["pass"] = self.passed
            result["total"] = self.count
            result["time"] = str(datetime.datetime.now())
            result["trackers"] = self.get_coordinates_dictionary()
            return result
