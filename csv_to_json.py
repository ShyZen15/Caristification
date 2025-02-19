import csv
import json
import os

def csv_to_json(file):
    if os.path.exists('data.json'):
        with open(file, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            data = [row for row in reader]

        with open('data.json', 'w') as jsonfile:
            json.dump(data, jsonfile)

