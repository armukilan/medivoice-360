


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Kaggle Hub is pre-installed in Kaggle notebooks
# ★ RUN THIS EVERY SESSION
import kagglehub

# Download Gemma 4 — this pulls directly from Kaggle's model registry
path = kagglehub.model_download("google/gemma-4/transformers/gemma-4-e4b-it")
print("Model path:", path)


# Upgrade transformers to latest version that supports Gemma 4
# ★ RUN THIS EVERY SESSION
import subprocess
subprocess.run(["pip", "install", "-q", "--upgrade", "transformers"], check=True)

# Restart kernel after this cell finishes
print("Done. Now go to Run menu → Restart Session, then run from Cell 2 onwards.")


# ★ RUN THIS EVERY SESSION
import torch
from transformers import AutoTokenizer, AutoModelForImageTextToText, AutoProcessor

MODEL_PATH = path  # from Cell 1

print("Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_PATH)

print("Loading model...")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()
print("Model loaded successfully.")


# ★ RUN THIS EVERY SESSION
def medivoice_generate(user_message, system_prompt=None, max_new_tokens=512):
    if system_prompt is None:
        system_prompt = """You are MediVoice, an AI clinical assistant for community 
health workers in rural and low-resource settings. You generate structured SOAP notes, 
detect the language of input automatically, flag urgency levels clearly, and always 
recommend referral for life-threatening conditions. Be concise and accurate."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [{"type": "text", "text": user_message}]}
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.3,
            do_sample=True,
        )

    input_len = inputs["input_ids"].shape[1]
    response = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
    return response





# EXCLUDE_FROM_PUSH
# Skip the cell
test_1 = """
Patient: 34-year-old woman
Symptoms: Fever 39.2C for 3 days, severe headache, stiff neck, sensitivity to light
Location: Rural village, no prior records available
Task: Generate a SOAP note and urgency level (Low / Medium / HIGH).
"""

print("=== TEST 1: English SOAP Note ===")
result = medivoice_generate(test_1)
print(result)


# EXCLUDE_FROM_PUSH
# Skip the cell
test_2 = """
Patient input (in Hindi): 
"मुझे तीन दिन से बुखार है, सिर में बहुत दर्द है और उल्टी आ रही है। मैं 28 साल का हूं।"

Task: 
1. Detect the language
2. Translate and understand the symptoms
3. Generate SOAP note in English
4. Also provide a summary in Hindi for the patient
5. Flag urgency level
"""

print("=== TEST 2: Hindi Input ===")
result = medivoice_generate(test_2)
print(result)


# EXCLUDE_FROM_PUSH
# Skip the cell
test_3 = """
Medicine label text extracted from photo:
Drug: Metformin Hydrochloride 500mg
Instructions: Take 1 tablet twice daily with meals
Warnings: Do not use if allergic to metformin. May cause stomach upset.
Patient's current medications: Aspirin 75mg daily, Lisinopril 10mg daily

Task:
1. Explain the medicine instructions in simple English
2. Also explain in Hindi (patient's language)
3. Check for dangerous drug interactions with current medications
4. Flag any warnings
"""

print("=== TEST 3: MedLabel Drug Check ===")
result = medivoice_generate(test_3)
print(result)


# EXCLUDE_FROM_PUSH
# Skip the cell
# Simulating a return visit — patient history passed in context
patient_history = """
PATIENT ID: MV-0042
Name: Priya Sharma | Age: 34 | Language: Hindi
Previous Visit (2025-04-01):
- Presented with fever, headache, stiff neck
- Diagnosed: Suspected bacterial meningitis
- Referred to district hospital immediately
- Urgency: HIGH
"""

test_4 = f"""
{patient_history}

Current Visit (2025-05-05):
Patient returns after hospital treatment. Reports feeling better.
New complaints: mild fatigue, slight dizziness when standing up.
Current meds: Ceftriaxone course completed, now on oral antibiotics Day 3.

Task:
1. Review history and note progress
2. Generate updated SOAP note in English
3. Provide patient summary in Hindi
4. Flag urgency level
"""

print("=== TEST 4: Return Visit with History ===")
result = medivoice_generate(test_4)
print(result)


# ★ RUN THIS EVERY SESSION
import json
import os
from datetime import datetime

# In Kaggle, we save to /kaggle/working/
DB_PATH = "/kaggle/working/medivoice_db.json"

def load_db():
    if os.path.exists(DB_PATH):
        with open(DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"patients": {}}

def save_db(db):
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)
    print(f"Database saved to {DB_PATH}")

def create_patient(patient_id, name, age, language):
    db = load_db()
    if patient_id in db["patients"]:
        print(f"Patient {patient_id} already exists.")
        return
    db["patients"][patient_id] = {
        "id": patient_id,
        "name": name,
        "age": age,
        "language": language,
        "visits": []
    }
    save_db(db)
    print(f"Patient {name} created with ID: {patient_id}")

def get_patient(patient_id):
    db = load_db()
    return db["patients"].get(patient_id, None)

def get_patient_history_text(patient_id):
    patient = get_patient(patient_id)
    if not patient:
        return None
    history = f"PATIENT: {patient['name']} | Age: {patient['age']} | Language: {patient['language']}\n"
    history += f"Total visits: {len(patient['visits'])}\n\n"
    for i, visit in enumerate(patient['visits']):
        history += f"--- Visit {i+1} ({visit['date']}) ---\n"
        history += f"{visit['soap_english']}\n"
        history += f"Urgency: {visit['urgency']}\n\n"
    return history

def save_visit(patient_id, soap_english, soap_regional, urgency, medications=None):
    db = load_db()
    if patient_id not in db["patients"]:
        print(f"Patient {patient_id} not found.")
        return
    visit = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "soap_english": soap_english,
        "soap_regional": soap_regional,
        "urgency": urgency,
        "medications": medications or []
    }
    db["patients"][patient_id]["visits"].append(visit)
    save_db(db)
    print(f"Visit saved for patient {patient_id}")

print("Patient database system ready.")


# ★ RUN THIS EVERY SESSION
def generate_soap_note(patient_id, consultation_text, language="Tamil"):
    """
    Full MediVoice pipeline:
    1. Pulls patient history from DB
    2. Generates bilingual SOAP note
    3. Saves back to DB
    4. Returns structured result
    """
    # Pull history if patient exists
    history = get_patient_history_text(patient_id)
    history_section = ""
    if history:
        history_section = f"""
PATIENT HISTORY FROM RECORDS:
{history}
---
"""

    prompt = f"""
{history_section}
CURRENT CONSULTATION:
{consultation_text}

Your task:
1. Generate a structured SOAP note in English with these exact sections:
   S (Subjective): What the patient reports
   O (Objective): Observations, vitals, measurements mentioned
   A (Assessment): Likely diagnosis or differential
   P (Plan): Treatment, referral, follow-up

2. After the SOAP note write:
   URGENCY: [LOW / MEDIUM / HIGH]
   REASON: [one line explanation]

3. Then write:
   PATIENT SUMMARY IN {language.upper()}:
   [Explain the situation and plan to the patient in simple {language}, 
   as if speaking directly to them. Use simple words, not medical jargon.]

Be concise. Be accurate. Always flag HIGH urgency if there are danger signs.
"""

    result = medivoice_generate(prompt, max_new_tokens=800)
    return result

print("SOAP generator ready.")


# ★ RUN THIS EVERY SESSION
def check_medications(drug_label_text, current_medications, language="Tamil"):
    """
    MedLabel module:
    Takes a drug label and checks against current medications.
    Explains in patient's language.
    """
    prompt = f"""
MEDICINE LABEL:
{drug_label_text}

PATIENT'S CURRENT MEDICATIONS:
{', '.join(current_medications) if current_medications else 'None reported'}

Your task:
1. Extract: Drug name, dosage, frequency, warnings
2. Check for dangerous interactions with current medications
3. Explain instructions in simple English
4. Explain instructions in simple {language} (as if speaking to the patient directly)
5. Flag any interactions or warnings clearly with WARNING: prefix

Be precise. Patient safety is critical.
"""
    result = medivoice_generate(prompt, max_new_tokens=600)
    return result

print("MedLabel module ready.")


# EXCLUDE_FROM_PUSH
# Skip the cell
# Step 1: Create a new patient
create_patient(
    patient_id="MV-0043",
    name="Kavitha Rajan",
    age=42,
    language="Tamil"
)

# Step 2: First consultation — input in Tamil
consultation = """
Patient says (in Tamil): 
"என்னுக்கு மூன்று நாளா காய்ச்சல் இருக்கு, தலை வலிக்குது, 
சாப்பிட்டா வாந்தி வருது. வயிறும் வலிக்குது."

(Translation context: Patient is a 42-year-old woman reporting 
3 days of fever, headache, vomiting after eating, and stomach pain.)

Vitals reported by health worker: Temp 38.8C, BP 110/70, Pulse 92
"""

print("=== FIRST VISIT: Kavitha Rajan ===\n")
result = generate_soap_note("MV-0043", consultation, language="Tamil")
print(result)


# EXCLUDE_FROM_PUSH
# Skip the cell
# Save the visit (manually parse urgency for now)
save_visit(
    patient_id="MV-0043",
    soap_english=result,  # full output for now
    soap_regional="See English record above — Tamil summary included",
    urgency="MEDIUM",  # update based on actual output
    medications=[]
)

# Simulate return visit 5 days later
return_visit = """
Patient returns after 5 days. Fever has subsided.
Still reports mild stomach pain and weakness.
Currently taking: Paracetamol 500mg as needed.
Health worker notes: Patient looks pale. Temp 37.1C, BP 108/68.
"""

print("=== RETURN VISIT: Kavitha Rajan ===\n")
result_2 = generate_soap_note("MV-0043", return_visit, language="Tamil")
print(result_2)





import subprocess
subprocess.run(["pip", "install", "-q", "gradio"], check=True)
print("Gradio installed.")


# EXCLUDE_FROM_PUSH
import gradio as gr

# ── Tab 1: New Consultation ──
def run_consultation(patient_id, consultation_text, language):
    if not patient_id or not consultation_text:
        return "Please enter both Patient ID and consultation text."
    patient = get_patient(patient_id)
    if not patient:
        return f"Patient ID {patient_id} not found. Please register the patient first."
    result = generate_soap_note(patient_id, consultation_text, language=language)
    # Auto-save visit
    save_visit(
        patient_id=patient_id,
        soap_english=result,
        soap_regional=f"Included in output above ({language} summary)",
        urgency="SEE OUTPUT",
        medications=[]
    )
    return result

# ── Tab 2: Patient Lookup ──
def lookup_patient(patient_id):
    if not patient_id:
        return "Please enter a Patient ID."
    patient = get_patient(patient_id)
    if not patient:
        return f"No patient found with ID: {patient_id}"
    history = get_patient_history_text(patient_id)
    summary = f"""
NAME: {patient['name']}
AGE: {patient['age']}
LANGUAGE: {patient['language']}
TOTAL VISITS: {len(patient['visits'])}

FULL HISTORY:
{history}
"""
    return summary

# ── Tab 3: MedLabel ──
def run_medlabel(drug_label_text, current_meds, language):
    if not drug_label_text:
        return "Please enter medicine label information."
    meds_list = [m.strip() for m in current_meds.split(",")] if current_meds else []
    result = check_medications(drug_label_text, meds_list, language=language)
    return result

# ── Tab 4: Register Patient ──
def register_patient(patient_id, name, age, language):
    if not all([patient_id, name, age, language]):
        return "Please fill in all fields."
    try:
        age = int(age)
    except ValueError:
        return "Age must be a number."
    existing = get_patient(patient_id)
    if existing:
        return f"Patient ID {patient_id} already exists: {existing['name']}"
    create_patient(patient_id, name, age, language)
    return f"Patient {name} registered successfully with ID: {patient_id}"

# ── Language choices ──
LANGUAGES = [
    "Tamil", "Hindi", "Telugu", "Kannada", "Malayalam",
    "Bengali", "Marathi", "Gujarati", "Punjabi",
    "Swahili", "Arabic", "French", "Spanish",
    "Portuguese", "Indonesian", "English"
]

# ── Build UI ──
with gr.Blocks(title="MediVoice 360", theme=gr.themes.Soft()) as app:

    gr.Markdown("""
    # MediVoice 360
    ### Offline-first · Multilingual · AI Clinical Assistant for Community Health Workers
    *Powered by Gemma 4 — runs fully without internet*
    """)

    with gr.Tabs():

        # Tab 1
        with gr.Tab("New Consultation"):
            gr.Markdown("Enter patient consultation details. MediVoice will generate a SOAP note in English and the patient's language.")
            with gr.Row():
                pid_consult = gr.Textbox(label="Patient ID", placeholder="e.g. MV-0043")
                lang_consult = gr.Dropdown(choices=LANGUAGES, value="Tamil", label="Patient Language")
            consult_input = gr.Textbox(
                label="Consultation Notes",
                placeholder="Describe symptoms, vitals, and observations. Can be in any language.",
                lines=6
            )
            consult_btn = gr.Button("Generate SOAP Note", variant="primary")
            consult_output = gr.Textbox(label="MediVoice Output", lines=20)
            consult_btn.click(
                fn=run_consultation,
                inputs=[pid_consult, consult_input, lang_consult],
                outputs=consult_output
            )

        # Tab 2
        with gr.Tab("Patient Lookup"):
            gr.Markdown("Pull up a patient's complete visit history.")
            pid_lookup = gr.Textbox(label="Patient ID", placeholder="e.g. MV-0043")
            lookup_btn = gr.Button("Lookup Patient", variant="primary")
            lookup_output = gr.Textbox(label="Patient History", lines=20)
            lookup_btn.click(
                fn=lookup_patient,
                inputs=pid_lookup,
                outputs=lookup_output
            )

        # Tab 3
        with gr.Tab("MedLabel — Drug Check"):
            gr.Markdown("Enter medicine label details to get instructions in the patient's language and check for drug interactions.")
            with gr.Row():
                lang_med = gr.Dropdown(choices=LANGUAGES, value="Tamil", label="Patient Language")
            drug_label = gr.Textbox(
                label="Medicine Label Text",
                placeholder="Drug name, dosage, frequency, warnings...",
                lines=4
            )
            current_meds = gr.Textbox(
                label="Current Medications (comma separated)",
                placeholder="e.g. Aspirin 75mg, Lisinopril 10mg"
            )
            med_btn = gr.Button("Check Medication", variant="primary")
            med_output = gr.Textbox(label="MedLabel Output", lines=15)
            med_btn.click(
                fn=run_medlabel,
                inputs=[drug_label, current_meds, lang_med],
                outputs=med_output
            )

        # Tab 4
        with gr.Tab("Register Patient"):
            gr.Markdown("Register a new patient in the system.")
            with gr.Row():
                reg_id = gr.Textbox(label="Patient ID", placeholder="e.g. MV-0044")
                reg_name = gr.Textbox(label="Full Name", placeholder="e.g. Anjali Devi")
            with gr.Row():
                reg_age = gr.Textbox(label="Age", placeholder="e.g. 35")
                reg_lang = gr.Dropdown(choices=LANGUAGES, value="Tamil", label="Primary Language")
            reg_btn = gr.Button("Register Patient", variant="primary")
            reg_output = gr.Textbox(label="Status")
            reg_btn.click(
                fn=register_patient,
                inputs=[reg_id, reg_name, reg_age, reg_lang],
                outputs=reg_output
            )

print("UI built. Ready to launch.")


# EXCLUDE_FROM_PUSH
app.launch(share=True)





# ★ RUN THIS EVERY SESSION
from PIL import Image
import requests
from io import BytesIO

def analyze_image(image, patient_id, language="Tamil"):
    """
    Takes an uploaded image (wound, rash, medicine label photo, 
    or medical document) and generates a clinical analysis.
    """
    if image is None:
        return "No image provided."

    patient = get_patient(patient_id) if patient_id else None
    patient_context = ""
    if patient:
        history = get_patient_history_text(patient_id)
        patient_context = f"""
PATIENT CONTEXT:
{history}
---
"""

    messages = [
        {
            "role": "system",
            "content": """You are MediVoice, an AI clinical assistant for 
community health workers. Analyze medical images carefully and provide 
structured clinical observations. Always recommend referral for serious findings."""
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image
                },
                {
                    "type": "text",
                    "text": f"""
{patient_context}
Please analyze this medical image and provide:

1. VISUAL FINDINGS: What do you observe in the image?
2. CLINICAL IMPRESSION: What condition(s) does this suggest?
3. SEVERITY: Mild / Moderate / Severe
4. RECOMMENDED ACTION: Treatment or referral guidance
5. URGENCY: LOW / MEDIUM / HIGH

Then provide a brief summary in {language} for the patient/health worker.

Important: If this is a medicine label, extract drug name, dosage, 
frequency and explain in {language}.
If this is a medical document (prescription, report), summarize key findings.
"""
                }
            ]
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=600,
            temperature=0.3,
            do_sample=True,
        )

    input_len = inputs["input_ids"].shape[1]
    response = processor.decode(
        outputs[0][input_len:],
        skip_special_tokens=True
    )
    return response

print("Image analysis function ready.")


# ★ RUN THIS EVERY SESSION
import gradio as gr

def run_image_analysis(image, patient_id, language):
    if image is None:
        return "Please upload an image."
    result = analyze_image(image, patient_id, language=language)
    if patient_id and get_patient(patient_id):
        save_visit(
            patient_id=patient_id,
            soap_english=f"[IMAGE ANALYSIS]\n{result}",
            soap_regional=f"Image analysis in {language} included above",
            urgency="SEE OUTPUT",
            medications=[]
        )
    return result

def run_recmed(drug_label_text, drug_image, current_meds, language):
    """
    RecMed — accepts either text OR image OR both.
    If image is provided, analyze it first, then combine with text input.
    """
    if not drug_label_text and drug_image is None:
        return "Please enter medicine details or upload an image."

    meds_list = [m.strip() for m in current_meds.split(",")] if current_meds else []

    # If image provided, analyze it first
    image_findings = ""
    if drug_image is not None:
        messages = [
            {
                "role": "system",
                "content": "You are MediVoice, an AI clinical assistant. Extract all text and information visible in this medicine label or medical image accurately."
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": drug_image},
                    {"type": "text", "text": "Extract all information from this image: drug name, dosage, frequency, warnings, ingredients, expiry date, and any other relevant details."}
                ]
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=400,
                temperature=0.2,
                do_sample=True,
            )
        input_len = inputs["input_ids"].shape[1]
        image_findings = processor.decode(
            outputs[0][input_len:],
            skip_special_tokens=True
        )

    # Combine image findings with any text input
    combined_label = ""
    if image_findings:
        combined_label += f"EXTRACTED FROM IMAGE:\n{image_findings}\n\n"
    if drug_label_text:
        combined_label += f"ADDITIONAL TEXT PROVIDED:\n{drug_label_text}"

    # Run full RecMed check
    result = check_medications(combined_label, meds_list, language=language)
    return result

# ── Language choices ──
LANGUAGES = [
    "Tamil", "Hindi", "Telugu", "Kannada", "Malayalam",
    "Bengali", "Marathi", "Gujarati", "Punjabi",
    "Swahili", "Arabic", "French", "Spanish",
    "Portuguese", "Indonesian", "English"
]

# ── Full UI ──
with gr.Blocks(title="MediVoice 360", theme=gr.themes.Soft()) as app:

    gr.Markdown("""
    # MediVoice 360
    ### Offline-first · Multilingual · AI Clinical Assistant for Community Health Workers
    *Powered by Gemma 4 — runs fully without internet*
    """)

    with gr.Tabs():

        # Tab 1 — New Consultation
        with gr.Tab("New Consultation"):
            gr.Markdown("Generate a bilingual SOAP note from consultation notes in any language.")
            with gr.Row():
                pid_consult = gr.Textbox(label="Patient ID", placeholder="e.g. MV-0043")
                lang_consult = gr.Dropdown(choices=LANGUAGES, value="Tamil", label="Patient Language")
            consult_input = gr.Textbox(
                label="Consultation Notes",
                placeholder="Describe symptoms, vitals, observations. Any language accepted.",
                lines=6
            )
            consult_btn = gr.Button("Generate SOAP Note", variant="primary")
            consult_output = gr.Textbox(label="MediVoice Output", lines=20)
            consult_btn.click(
                fn=run_consultation,
                inputs=[pid_consult, consult_input, lang_consult],
                outputs=consult_output
            )

        # Tab 2 — Image Analysis
        with gr.Tab("Image Analysis"):
            gr.Markdown("""
            Upload any medical image — wound, rash, ECG, EEG, MRI, CT scan, 
            blood report, urine test, prescription, or any medical document. 
            MediVoice analyzes it and explains findings in the patient's language.
            """)
            with gr.Row():
                pid_image = gr.Textbox(label="Patient ID (optional)", placeholder="e.g. MV-0043")
                lang_image = gr.Dropdown(choices=LANGUAGES, value="Tamil", label="Patient Language")
            image_input = gr.Image(label="Upload Medical Image", type="pil")
            image_btn = gr.Button("Analyze Image", variant="primary")
            image_output = gr.Textbox(label="Analysis Output", lines=20)
            image_btn.click(
                fn=run_image_analysis,
                inputs=[image_input, pid_image, lang_image],
                outputs=image_output
            )

        # Tab 3 — Patient Lookup
        with gr.Tab("Patient Lookup"):
            gr.Markdown("Pull up a patient's complete visit history.")
            pid_lookup = gr.Textbox(label="Patient ID", placeholder="e.g. MV-0043")
            lookup_btn = gr.Button("Lookup Patient", variant="primary")
            lookup_output = gr.Textbox(label="Patient History", lines=20)
            lookup_btn.click(
                fn=lookup_patient,
                inputs=pid_lookup,
                outputs=lookup_output
            )

        # Tab 4 — RecMed
        with gr.Tab("RecMed — Medicine Check"):
            gr.Markdown("""
            Upload a photo of a medicine bottle, blister pack, or prescription — 
            OR type the medicine details manually. RecMed explains instructions 
            in the patient's language and checks for dangerous drug interactions.
            """)
            with gr.Row():
                lang_med = gr.Dropdown(choices=LANGUAGES, value="Tamil", label="Patient Language")
            drug_image_input = gr.Image(label="Upload Medicine Label / Prescription (optional)", type="pil")
            drug_label = gr.Textbox(
                label="Medicine Details (optional — text input)",
                placeholder="Drug name, dosage, frequency, warnings...",
                lines=3
            )
            current_meds = gr.Textbox(
                label="Current Medications (comma separated)",
                placeholder="e.g. Aspirin 75mg, Lisinopril 10mg"
            )
            med_btn = gr.Button("Check Medicine", variant="primary")
            med_output = gr.Textbox(label="RecMed Output", lines=15)
            med_btn.click(
                fn=run_recmed,
                inputs=[drug_label, drug_image_input, current_meds, lang_med],
                outputs=med_output
            )

        # Tab 5 — Register Patient
        with gr.Tab("Register Patient"):
            gr.Markdown("Register a new patient in the local database.")
            with gr.Row():
                reg_id = gr.Textbox(label="Patient ID", placeholder="e.g. MV-0044")
                reg_name = gr.Textbox(label="Full Name", placeholder="e.g. Anjali Devi")
            with gr.Row():
                reg_age = gr.Textbox(label="Age", placeholder="e.g. 35")
                reg_lang = gr.Dropdown(choices=LANGUAGES, value="Tamil", label="Primary Language")
            reg_btn = gr.Button("Register Patient", variant="primary")
            reg_output = gr.Textbox(label="Status")
            reg_btn.click(
                fn=register_patient,
                inputs=[reg_id, reg_name, reg_age, reg_lang],
                outputs=reg_output
            )

print("Full UI with RecMed and image support ready.")


# ★ RUN THIS EVERY SESSION
app.launch(share=True)


# EXCLUDE_FROM_PUSH
import os
import subprocess
from kaggle_secrets import UserSecretsClient

# ── Fetch token securely ──
secrets = UserSecretsClient()
GITHUB_TOKEN = secrets.get_secret("GITHUB_TOKEN")

# ── CONFIG ──
GITHUB_USERNAME = "your_github_username"
REPO_NAME = "medivoice-360"
COMMIT_MESSAGE = "Update MediVoice 360"

# ── Find the notebook file ──
notebook_dir = "/kaggle/input"
working_dir = "/kaggle/working"

# List all notebooks available
print("Looking for notebooks...")
for root, dirs, files in os.walk("/kaggle"):
    for f in files:
        if f.endswith(".ipynb"):
            print(os.path.join(root, f))


# EXCLUDE_FROM_PUSH
import os
import shutil
import subprocess
from kaggle_secrets import UserSecretsClient

# ── Fetch token securely ──
secrets = UserSecretsClient()
GITHUB_TOKEN = secrets.get_secret("GITHUB_TOKEN")

# ── CONFIG ──
GITHUB_USERNAME = "armukilan"
REPO_NAME = "medivoice-360"
COMMIT_MESSAGE = "Update MediVoice 360"

# ── The virtual document is already a Python script ──
source_path = "/kaggle/working/.virtual_documents/__notebook_source__.ipynb"
output_py = "/kaggle/working/medivoice360.py"

# Just copy and rename it as .py
shutil.copy(source_path, output_py)
print(f"Copied to {output_py}")

# ── Push to GitHub ──
repo_url = f"https://{GITHUB_USERNAME}:{GITHUB_TOKEN}@github.com/{GITHUB_USERNAME}/{REPO_NAME}.git"
work_dir = "/kaggle/working/repo"

if os.path.exists(work_dir):
    subprocess.run(["rm", "-rf", work_dir])

subprocess.run(["git", "clone", repo_url, work_dir], check=True)
subprocess.run(["cp", output_py, f"{work_dir}/medivoice360.py"], check=True)

subprocess.run(["git", "config", "user.email", "you@example.com"], cwd=work_dir)
subprocess.run(["git", "config", "user.name", GITHUB_USERNAME], cwd=work_dir)

subprocess.run(["git", "add", "."], cwd=work_dir, check=True)
subprocess.run(["git", "commit", "-m", COMMIT_MESSAGE], cwd=work_dir, check=True)
subprocess.run(["git", "push", "origin", "main"], cwd=work_dir, check=True)

print("Successfully pushed to GitHub!")
print(f"https://github.com/{GITHUB_USERNAME}/{REPO_NAME}")
