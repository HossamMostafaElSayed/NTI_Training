# src/models/patient_model.py

from src.utils.api_client import api_request
from datetime import date

def get_all_patients(doctor_id=None):
    params = {'doctor_id': doctor_id} if doctor_id else {}
    response = api_request("get", "patients", params=params)
    return response if response is not None else []

def search_patients(query, doctor_id=None):
    """Searches for patients via the API using a single query term."""
    params = {"query": query}
    if doctor_id:
        params['doctor_id'] = doctor_id
    response = api_request("get", "patients/search", params=params)
    return response if response is not None else []

def add_patient(name, phone_number, birth_date, gender, address, email, emergency_contact_name, emergency_contact_relationship, emergency_contact_phone, doctor_id):
    data = {
        "name": name, "phone_number": phone_number, "birth_date": birth_date, "gender": gender,
        "address": address, "email": email, "emergency_contact_name": emergency_contact_name,
        "emergency_contact_relationship": emergency_contact_relationship,
        "emergency_contact_phone": emergency_contact_phone, "doctor_id": doctor_id
    }
    return api_request("post", "patients", json=data)
def save_obstetrics_data(patient_id: int, lmp_date: date, edd_date: date):
    """Saves the LMP and EDD for a specific patient via the API."""
    data = {"lmp_date": lmp_date.isoformat(), "edd_date": edd_date.isoformat()}
    return api_request("put", f"patients/{patient_id}/obstetrics", json=data)
def get_patient_by_id(patient_id):
    return api_request("get", f"patients/{patient_id}")

def update_patient_details(patient_id, name, phone_number, birth_date, gender, address, email, emergency_contact_name, emergency_contact_relationship, emergency_contact_phone):
    data = {
        "name": name, "phone_number": phone_number, "birth_date": birth_date, "gender": gender,
        "address": address, "email": email, "emergency_contact_name": emergency_contact_name,
        "emergency_contact_relationship": emergency_contact_relationship,
        "emergency_contact_phone": emergency_contact_phone
    }
    return api_request("put", f"patients/{patient_id}", json=data)

def get_patient_diagnoses(patient_id):
    response = api_request("get", f"diagnoses/patient/{patient_id}")
    return response if response is not None else []

def add_diagnosis(patient_id, data):
    data['patient_id'] = patient_id
    return api_request("post", "diagnoses", json=data)

def update_diagnosis(diagnosis_id, data):
    return api_request("put", f"diagnoses/{diagnosis_id}", json=data)

def delete_diagnosis(diagnosis_id):
    return api_request("delete", f"diagnoses/{diagnosis_id}")

def get_patient_notes(patient_id):
    """
    Retrieves the 'notes' from the most recent visit for a patient.
    This is used by the AddNotesDialog.
    """
    # This calls a new API endpoint we will create on the backend.
    return api_request("get", f"patients/{patient_id}/latest-note")