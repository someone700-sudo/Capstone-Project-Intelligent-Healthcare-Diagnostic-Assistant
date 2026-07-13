# ============================================================  
# CAPSTONE MAIN APPLICATION  
# Intelligent Healthcare Diagnostic Assistant  
# Introduction to AI — 13-Week Capstone  
# ============================================================  

import sys  
import json  
import warnings  
import numpy as np  
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for chart generation
import matplotlib.pyplot as plt  
import matplotlib.gridspec as gridspec  
warnings.filterwarnings('ignore')  

# Fix Windows console encoding for emoji/unicode output
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Import all modules  
from modules.agent          import HealthcareDiagnosticAgent, PatientPercept  
from modules.knowledge_base import MedicalKnowledgeBase  
from modules.bayesian_net   import SimpleBayesianDiagnostics  
from modules.ml_classifier  import MLDiagnosticClassifier  
from modules.neural_network import NeuralDiagnosticModel  
from modules.fuzzy_controller import FuzzySeverityAssessor  
from modules.planner        import TreatmentPlanner  

# ── ANSI Colors ────────────────────────────────────────────  
class C:  
    HEADER = '\033[95m'; BLUE   = '\033[94m'  
    GREEN  = '\033[92m'; YELLOW = '\033[93m'  
    RED    = '\033[91m'; BOLD   = '\033[1m'  
    END    = '\033[0m'  

def banner():  
    print(f"""  
{C.BOLD}{C.BLUE}  
╔══════════════════════════════════════════════════════════╗  
║        🏥 INTELLIGENT HEALTHCARE DIAGNOSTIC AI           ║  
║         Introduction to AI — Capstone Project            ║  
║  Modules: Agents | Logic | Bayes | ML | DNN | Fuzzy      ║  
╚══════════════════════════════════════════════════════════╝  
{C.END}""")  

def section(title: str):  
    print(f"\n{C.BOLD}{C.YELLOW}{'═'*60}{C.END}")  
    print(f"{C.BOLD}{C.YELLOW}  {title}{C.END}")  
    print(f"{C.BOLD}{C.YELLOW}{'═'*60}{C.END}")  

def build_system() -> HealthcareDiagnosticAgent:  
    """Instantiate and wire all AI modules"""  
    section("🔧 Building AI System — Registering Modules")  

    agent = HealthcareDiagnosticAgent()  

    print("\n  Initializing modules...")  
    modules = {  
        'KnowledgeBase': MedicalKnowledgeBase(),  
        'BayesianNet':   SimpleBayesianDiagnostics(),  
        'MLClassifier':  MLDiagnosticClassifier(),  
        'NeuralNetwork': NeuralDiagnosticModel(),  
        'FuzzyAssessor': FuzzySeverityAssessor(),  
        'Planner':       TreatmentPlanner(),  
    }  

    for name, mod in modules.items():  
        agent.register_module(name, mod)  

    print(f"\n  ✅ All {len(modules)} modules registered successfully!")  
    return agent  


def get_test_patients():  
    """Define 5 diverse test patients covering different diseases"""  
    return [  
        PatientPercept(  
            patient_id="P001",  
            symptoms=["fever", "cough", "fatigue", "loss_of_smell", "headache"],  
            age=34,  
            temperature=38.5,  
            heart_rate=92,  
            blood_pressure="130/85"  
        ),  
        PatientPercept(  
            patient_id="P002",  
            symptoms=["chest_pain", "shortness_of_breath", "sweating", "fatigue"],  
            age=58,  
            temperature=37.2,  
            heart_rate=125,  
            blood_pressure="160/100"  
        ),  
        PatientPercept(  
            patient_id="P003",  
            symptoms=["fever", "rash", "joint_pain", "headache", "fatigue", "body_aches"],  
            age=27,  
            temperature=39.8,  
            heart_rate=108,  
            blood_pressure="120/75"  
        ),  
        PatientPercept(  
            patient_id="P004",  
            symptoms=["fever", "cough", "fatigue", "body_aches", "headache"],  
            age=42,  
            temperature=39.0,  
            heart_rate=95,  
            blood_pressure="125/80"  
        ),  
        PatientPercept(  
            patient_id="P005",  
            symptoms=["frequent_urination", "excessive_thirst", "blurred_vision", "fatigue"],  
            age=51,  
            temperature=37.0,  
            heart_rate=78,  
            blood_pressure="140/90"  
        ),  
    ]  


def print_diagnosis_report(report: dict, patient: PatientPercept):  
    """Pretty-print a single patient's diagnosis report"""  
    print(f"\n{C.BOLD}{C.GREEN}┌──────────────────────────────────────────────────┐{C.END}")  
    print(f"{C.BOLD}{C.GREEN}│  Patient: {report['patient_id']:<39}│{C.END}")  
    print(f"{C.BOLD}{C.GREEN}└──────────────────────────────────────────────────┘{C.END}")  

    print(f"  🩺 Symptoms:       {', '.join(report['symptoms'])}")  
    print(f"  🌡️  Temperature:    {patient.temperature}°C")  
    print(f"  💓 Heart Rate:     {patient.heart_rate} bpm")  
    print(f"  🩸 Blood Pressure: {patient.blood_pressure}")  
    print(f"  📅 Age:            {patient.age}")  

    # Diagnosis  
    urgency_colors = {  
        "CRITICAL": C.RED,  
        "HIGH":     C.YELLOW,  
        "MEDIUM":   C.BLUE,  
        "LOW":      C.GREEN  
    }  
    uc = urgency_colors.get(report['urgency'], C.END)  

    print(f"\n  {C.BOLD}── Diagnosis ──{C.END}")  
    print(f"  🔬 Diagnosis:   {C.BOLD}{report['diagnosis']}{C.END}")  
    print(f"  📊 Confidence:  {report['confidence']:.1%}")  
    print(f"  ⚠️  Urgency:     {uc}{C.BOLD}{report['urgency']}{C.END}")  

    # Recommendations  
    print(f"\n  {C.BOLD}── Recommendations ──{C.END}")  
    for rec in report['recommendations']:  
        print(f"  {rec}")  

    print(f"  📋 Next Action: {report['next_action']}")  
    print(f"  {'─'*50}")  


def print_treatment_plan(planner: TreatmentPlanner, diagnosis: str, urgency: str):  
    """Generate and print a treatment plan"""  
    plan_result = planner.create_treatment_plan(diagnosis, urgency)  

    if 'error' in plan_result:  
        print(f"  ⚠️  {plan_result['error']}")  
        return  

    print(f"\n  {C.BOLD}{C.BLUE}── Treatment Plan ({diagnosis} | {urgency}) ──{C.END}")  
    print(f"  Initial State: {', '.join(plan_result['initial_state'])}")  
    print(f"  Goal State:    {', '.join(plan_result['goal_state'])}")  
    print(f"  Total Steps:   {plan_result['steps']}")  
    print()  

    for step in plan_result['plan']:  
        print(f"    Step {step['step']}: {C.BOLD}{step['action']}{C.END} "  
              f"(⏱ {step['duration']})")  

    print(f"\n  ⏱ Duration: {plan_result['total_duration']}")  


def run_evaluation(ml_classifier: MLDiagnosticClassifier):  
    """Run the evaluation module to generate charts"""  
    try:  
        from evaluation.metrics import ModelEvaluator  
        section("📊 Model Evaluation — Generating Charts")  
        evaluator = ModelEvaluator(ml_classifier)  
        evaluator.run_full_evaluation()  
    except ImportError:  
        print("  ⚠️  Evaluation module not found — skipping charts.")  


# ── MAIN ────────────────────────────────────────────────────  
def main():  
    banner()  

    # ─── Step 1: Build the AI System ───  
    agent = build_system()  

    # ─── Step 2: Train ML Models ───  
    section("🧠 Training Machine Learning Models")  
    ml_classifier = agent._modules['MLClassifier']  
    ml_results = ml_classifier.train(verbose=True)  

    # ─── Step 3: Train Neural Network ───  
    section("🔮 Training Deep Neural Network")  
    nn_model = agent._modules['NeuralNetwork']  
    nn_results = nn_model.train(epochs=50, verbose=1)  

    # ─── Step 4: Diagnose Test Patients ───  
    section("🏥 Running Diagnostic Pipeline — 5 Test Patients")  
    patients = get_test_patients()  
    all_reports = []  

    for i, patient in enumerate(patients, 1):  
        print(f"\n{'━'*60}")  
        print(f"  Processing Patient {i}/{len(patients)}: {patient.patient_id}")  
        print(f"{'━'*60}")  

        # Run the full agent cycle: Perceive → Think → Act  
        report = agent.run(patient)  
        all_reports.append(report)  

        # Print the diagnosis report  
        print_diagnosis_report(report, patient)  

        # Generate and print treatment plan  
        planner = agent._modules['Planner']  
        print_treatment_plan(planner, report['diagnosis'], report['urgency'])  

        # Print fuzzy severity assessment  
        fuzzy = agent._modules['FuzzyAssessor']  
        severity = fuzzy.assess(  
            patient.temperature, patient.heart_rate, len(patient.symptoms)  
        )  
        print(f"\n  {C.BOLD}── Fuzzy Severity Assessment ──{C.END}")  
        print(f"  🎯 Score: {severity['severity_score']}/100")  
        print(f"  📋 Label: {severity['severity_label']}")  
        print(f"  📊 Rule Strengths: {json.dumps(severity['rule_strengths'], indent=6)}")  

    # ─── Step 5: Agent Performance Summary ───  
    section("📈 Agent Performance Summary")  
    perf = agent.get_performance()  
    print(f"  Total Patients Processed: {perf['total_patients']}")  
    print(f"  Diagnoses Made:           {perf['diagnoses_made']}")  
    print(f"  Performance Score:        {perf['performance_score']}")  
    agent.print_log()  

    # ─── Step 6: Diagnosis Summary Table ───  
    section("📋 Diagnosis Summary — All Patients")  
    print(f"\n  {'Patient':<10} {'Diagnosis':<20} {'Confidence':<12} {'Urgency':<10}")  
    print(f"  {'─'*52}")  
    for report in all_reports:  
        print(f"  {report['patient_id']:<10} "  
              f"{report['diagnosis']:<20} "  
              f"{report['confidence']:<12.1%} "  
              f"{report['urgency']:<10}")  

    # ─── Step 7: ML Evaluation Charts ───  
    section("📊 Generating Evaluation Charts")  
    print("\n  Generating ML evaluation plots...")  
    ml_classifier.plot_evaluation()  

    print("\n  Generating Neural Network training curves...")  
    nn_model.plot_training()  

    # ─── Step 8: Full Evaluation Module ───  
    run_evaluation(ml_classifier)  

    # ─── Done ───  
    section("✅ Capstone Project Complete!")  
    print(f"""  
  {C.BOLD}{C.GREEN}All modules executed successfully!{C.END}  
  
  Modules Used:  
    1. Intelligent Agent (Model-Based + Goal-Based)  
    2. Knowledge Base (Forward & Backward Chaining)  
    3. Bayesian Network (Naïve Bayes Posterior)  
    4. ML Classifier (Decision Tree, Random Forest, Gradient Boosting)  
    5. Deep Neural Network (MLP with Dropout & BatchNorm)  
    6. Fuzzy Logic (Severity Assessment)  
    7. AI Planner (BFS Treatment Planning)  
    
  Patients Diagnosed: {len(all_reports)}  
  Charts Generated: ml_evaluation.png, nn_training.png  
""")  


if __name__ == '__main__':  
    main()  
