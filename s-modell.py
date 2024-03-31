import spacy
from collections import defaultdict
from spacy.lang.en.stop_words import STOP_WORDS
import pandas as pd

# Laden des Spacy-Modell
nlp = spacy.load('en_core_web_md')

with open('C:\PythonProjects\Digiarc\s12.txt', 'r') as file:
    songs = file.read()

# Anwendung des Models
doc = nlp(songs)

# Lemmatisierung und Entfernung von Stoppwörtern
lemmatized_text = [token.lemma_ for token in doc if token.lemma_ not in STOP_WORDS]
#print(lemmatized_text)

# Erstellen Sie ein Dictionary zur Speicherung der Häufigkeit jeder Entity in jeder Kategorie
entity_freq = defaultdict(lambda: defaultdict(int))

# Durchlaufen für jede erkannte Entität
for ent in doc.ents:
    
    # Wenn die Entität ein Eigenname ist, erhöht sich die Häufigkeit im Dictionary
    if ent.label_ in ['PERSON', 'ORG', 'LOC', 'GPE', 'MISC']:
        entity_freq[ent.label_][ent.text] += 1
       

# Korrekturdatensatz
correction_data_df = pd.read_excel('correction_data.xlsx', dtype={'frequency': int})

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

def missing_person_percentage(correction_data, missing_entities):
    # Anzahl der 'PERSON'-Entities im Korrekturdatensatz
    person_count = sum(freq for label, _, freq in correction_data if label == 'PERSON')

    # Anzahl der 'PERSON'-Entities in den fehlenden Entities
    missing_person_count = sum(1 for label, _ in missing_entities if label == 'PERSON')

    # Berechnung des Prozentsatzes
    percentage = (missing_person_count / person_count) * 100

    return percentage

percentage = missing_person_percentage(correction_data, missing)
print(f'The percentage of PERSON entities in the correction data that were missing is {percentage}%.')

# Die Gesamtzahl der korrekten Entities
total_correct = sum(freq for _, _, freq in correction_data)

# Die Gesamtzahl der erkannten Entities
total_detected = sum(freq for entities in entity_freq.values() for freq in entities.values())

# Die Anzahl der korrekten Entities
total_correct1 = sum(entity_freq[label][entity] for label, entity, _ in correction_data)

# Den Prozentsatz der falschen Entities
percentage_false = (total_detected / total_correct) * 100

print(f'The percentage of false detected entities is {percentage_false-100}%.')

# Die Gesamtfehleranzahl
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

# Den Prozentsatzfehler
percentage_error = (total_error / total_correct) * 100

print(f'The total accuracy is {100-percentage_error}%.')

#spacey erkennt kein loc, oder misc


