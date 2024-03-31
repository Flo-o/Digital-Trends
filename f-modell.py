from flair.data import Sentence
from flair.models import SequenceTagger
from collections import defaultdict
import pandas as pd

# Laden des vortrainierten Modell
tagger = SequenceTagger.load('ner')

with open('C:\PythonProjects\Digiarc\songtexte.txt', 'r') as file:
    songs = file.read()

# Sentence Objekt erstellen
sentence = Sentence(songs)

# Model ausführen
tagger.predict(sentence)

# Erstellen des Dictionaries für die freq
entity_freq = defaultdict(lambda: defaultdict(int))

# Durchlaufen der erkannten Entität
for entity in sentence.get_spans('ner'):
    if entity.tag in ['MISC', 'LOC', 'PER', 'ORG']:
        entity_freq[entity.tag][entity.text] += 1

# Korrekturdatensatz
# Korrekturdatensatz
correction_data_df = pd.read_excel('correction_dataf.xlsx', dtype={'frequency': int})

# Konvertieren des DataFrames in eine Liste von Tupeln
correction_data = list(correction_data_df.itertuples(index=False, name=None))
def missing_entities(correction_data, entity_freq):
    missing_entities = []

    for label, entity, _ in correction_data:
        if entity_freq[label][entity] == 0:
            missing_entities.append((label, entity))

    return missing_entities

def extra_entities(correction_data, entity_freq):
    # Erstellen eines Sets aller Entities im Korrekturdatensatz
    correction_entities = set(entity for label, entity, _ in correction_data)

    # Erstellen eines Sets aller erkannten Entities
    detected_entities = set(entity for entities in entity_freq.values() for entity in entities.keys())

    # Finden der Entities, die vom Modell erkannt wurden, aber nicht im Korrekturdatensatz stehen
    extra_entities = detected_entities - correction_entities

    return extra_entities

extra = extra_entities(correction_data, entity_freq)
print(extra)


missing = missing_entities(correction_data, entity_freq)
print(missing)

# korrekte Entities
total_correct = sum(freq for _, _, freq in correction_data)

# Erkannte Entities
total_detected = sum(freq for entities in entity_freq.values() for freq in entities.values())

# Anzahl der korrekten Entities
total_correct1 = sum(entity_freq[label][entity] for label, entity, _ in correction_data)

# Prozentsatz der falschen Entities
percentage_false = (total_detected / total_correct) * 100

print(f'The percentage of false detected entities is {percentage_false-100}%.')

# Berechnen der Gesamtfehleranzahl
total_error = 0
total_error1 = 0
for label, entity, correct_freq in correction_data:
    detected_freq = entity_freq[label][entity]
    if detected_freq > 0 and label != entity_freq[label][entity]:
        total_error1 += abs(detected_freq - correct_freq)
    else:
        total_error += abs(detected_freq - correct_freq)
print(total_error1)        
total_error = total_error + (total_error1*0.5)
# Berechnen des Prozentsatzfehler
percentage_error = (total_error / total_correct) * 100

print(f'The total accuracy is {100-percentage_error}%.')
