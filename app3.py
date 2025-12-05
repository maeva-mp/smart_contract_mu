##AI-Powered Legal Contract Analysis System (for Mauritius)
# Imports
import os
import io
import math
import json
import pandas as pd
import numpy as np
import altair as alt
import streamlit as st

# Set page config
st.set_page_config(page_title="SmartContract MU", layout="wide", page_icon="⚖️")

# Set cache directories 
from pathlib import Path
CACHE_DIR = Path.home() / ".cache" / "huggingface"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ['TRANSFORMERS_CACHE'] = str(CACHE_DIR)
os.environ['SENTENCE_TRANSFORMERS_HOME'] = str(CACHE_DIR)
os.environ['HF_HOME'] = str(CACHE_DIR)

# Imports
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta
from typing import List, Tuple, Dict
from PyPDF2 import PdfReader
import docx
from sentence_transformers import SentenceTransformer
import faiss
from transformers import T5ForConditionalGeneration, T5TokenizerFast
import re
from rouge_score import rouge_scorer
import spacy
from spacy.pipeline import EntityRuler
import matplotlib.pyplot as plt

# Configuration
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SUMMARIZER_MODEL_NAME = "t5-small"
CHUNK_TOKENS = 300
EMBED_BATCH = 32
TOP_K = 5
MAX_NLP_CHARS = 100000
MIN_LINE_LENGTH = 3
MAX_OBLIGATIONS_DISPLAY = 20
MAX_ENTITIES_DISPLAY = 15
MAX_RISK_KEYWORDS_DISPLAY = 10
THRESHOLD = 0.5
COLOR_OPACITY = "33"
FAISS_INDEX_FILE = "faiss.index"
METADATA_FILE = "metadata.txt"
METRICS_FILE = "metrics.json"
GROUND_TRUTH_FILE = "ground_truth.json"

# Initialize session state for lite mode
if 'LITE_MODE' not in st.session_state:
    st.session_state.LITE_MODE = False

#reads text cases from a JSON file/writes new ones to that file
def load_ground_truth(path: str = GROUND_TRUTH_FILE) -> Dict:
    """Load ground truth query-document pairs"""
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        st.warning(f"⚠️ Could not load ground truth: {e}")
    return {}

def save_ground_truth(ground_truth: Dict, path: str = GROUND_TRUTH_FILE):
    """Save ground truth query-document pairs"""
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(ground_truth, f, indent=2)
    except IOError as e:
        st.error(f"❌ Could not save ground truth: {e}")

# backward compatibility
load_ground_truth_cases = load_ground_truth
save_ground_truth_cases = save_ground_truth

# Contract-specific risk keywords
RISK_KEYWORDS = {
    "critical": [
        "unlimited liability",
        "jointly and severally liable",
        "personal guarantee",
        "indemnify and hold harmless",
        "waive all rights",
        "waiver of rights",
        "terminate without cause",
        "unilateral termination",
        "terminate immediately without notice",
        "exclusive remedy",
        "sole remedy",
        "liquidated damages",
        "penalty clause",
        "penalty of",
        "unconditionally and irrevocably",
        "without first pursuing",
        "unlimited and joint"
    ],
    "high": [
        "indemnify",
        "indemnification",
        "hold harmless",
        "limitation of liability",
        "liable for damages",
        "breach of contract",
        "event of default",
        "material breach",
        "warranty disclaimer",
        "as is",
        "non-compete",
        "non-solicitation",
        "confidential information",
        "proprietary information",
        "intellectual property rights",
        "binding arbitration",
        "waive any right"
    ],
    "medium": [
        "termination notice",
        "notice period",
        "payment terms",
        "late payment",
        "force majeure",
        "renewal",
        "amendment",
        "governing law",
        "dispute resolution",
        "confidentiality",
        "delivery deadline",
        "milestone"
    ]
}

#Mauritius-specific compliance patterns 
MAURITIUS_COMPLIANCE = {
    "workers_rights_act": [
        "employment contract", "written particulars", "14 days",
        "workers rights", "termination notice", "severance"
    ],
    "civil_code": [
        "article 1108", "consent", "capacity", "object", "cause",
        "potestative", "three year", "limitation period"
    ],
    "data_protection": [
        "personal data", "data protection", "gdpr", "consent",
        "data subject", "processing"
    ]
}

# Mauritius Civil Code Articles Reference (Enhanced)
CIVIL_CODE_ARTICLES = {
    "1108": {
        "title": "Essential Conditions for Contract Validity",
        "text": "Four requisites are essential for the validity of an agreement: (1) Consent of the party who binds himself; (2) Capacity to contract; (3) A determinate object forming the subject-matter of the undertaking; (4) A lawful cause in the obligation",
        "keywords": ["consent", "capacity", "object", "cause", "validity", "essential conditions"],
        "risk_level": "CRITICAL",
        "explanation": "Missing any of these 4 elements makes the contract VOID"
    },
    "1131": {
        "title": "Obligation Without Cause",
        "text": "An obligation without a cause, or founded upon a false or unlawful cause, has no effect",
        "keywords": ["no cause", "false cause", "unlawful cause", "void", "invalid"],
        "risk_level": "CRITICAL",
        "explanation": "Contract must have a real, lawful reason for existing"
    },
    "1132": {
        "title": "Lawful Cause Required",
        "text": "The cause is unlawful when it is prohibited by law, or when it is contrary to good morals or to public order",
        "keywords": ["unlawful", "prohibited", "immoral", "public order", "illegal purpose"],
        "risk_level": "CRITICAL",
        "explanation": "Cannot contract for illegal or immoral purposes"
    },
    "1133": {
        "title": "Undeclared Cause",
        "text": "The cause need not be expressed. It is presumed to exist until the contrary is proved",
        "keywords": ["undeclared", "presumed", "implied cause", "burden of proof"],
        "risk_level": "LOW",
        "explanation": "Cause doesn't need to be written, but must exist"
    },
    "1134": {
        "title": "Force of Agreements (Binding Effect)",
        "text": "Agreements lawfully entered into have the force of law between the contracting parties",
        "keywords": ["binding", "force of law", "legally binding", "pacta sunt servanda"],
        "risk_level": "HIGH",
        "explanation": "Valid contracts are as binding as the law itself"
    },
    "1147": {
        "title": "Breach and Damages",
        "text": "A debtor shall be condemned, where appropriate, to pay damages for non-performance of the obligation",
        "keywords": ["breach", "damages", "non-performance", "compensation", "liability"],
        "risk_level": "HIGH",
        "explanation": "Failing to perform contract obligations = must pay damages"
    },
    "1174": {
        "title": "Potestative Condition (Void)",
        "text": "Any obligation is void when it has been contracted under a condition which is purely potestative on the part of the party who binds himself",
        "keywords": ["potestative", "purely discretionary", "void condition", "illusory"],
        "risk_level": "CRITICAL",
        "explanation": "Cannot make obligation depend solely on one party's whim"
    },
    "1184": {
        "title": "Resolutory Condition (Right to Cancel)",
        "text": "A resolutory condition is always implied in synallagmatic contracts for the case where one of the two parties does not perform his undertaking",
        "keywords": ["termination", "breach", "mutual obligations", "rescind", "cancel"],
        "risk_level": "HIGH",
        "explanation": "Automatic right to cancel if other party breaches"
    },
    "2262": {
        "title": "General Prescription Period (30 Years)",
        "text": "All personal and real actions are prescribed by thirty years, without the person who alleges such prescription being obliged to produce a title, or that the lapse of time may be opposed to him as a bar",
        "keywords": ["prescription", "limitation", "30 years", "time bar", "statutory period"],
        "risk_level": "MEDIUM",
        "explanation": "Claims expire after 30 years unless shorter period applies"
    },
    "2277": {
        "title": "Short Prescription (3 Years for Payment)",
        "text": "Actions of innkeepers and traiteurs for lodging and food; those of workmen and labourers for their work; those of schoolmasters and teachers for tuition and board; actions for payment of salaries, wages, pensions and annuities are prescribed by three years",
        "keywords": ["three years", "3 years", "payment", "wages", "professional fees", "salaries", "prescription"],
        "risk_level": "HIGH",
        "explanation": "Claims for payment/wages expire after only 3 years!"
    }
}

#Workers' Rights Act 2019 - Key Provisions 
WORKERS_RIGHTS_PROVISIONS = {
    "Section 27": {
        "title": "Written Particulars of Employment",
        "text": "Employer must provide written particulars within 14 days",
        "keywords": ["written particulars", "14 days", "employment terms"]
    },
    "Section 36": {
        "title": "Termination Notice Periods",
        "text": "Notice periods for termination of employment contracts",
        "keywords": ["termination notice", "notice period", "dismissal"]
    },
    "Section 91": {
        "title": "Severance Allowance",
        "text": "Entitlement to severance allowance on termination",
        "keywords": ["severance", "allowance", "termination pay"]
    }
}

#Data Protection Act 2017
DATA_PROTECTION_ARTICLES = {
    "Section 5": {
        "title": "Principles of Data Processing",
        "text": "Personal data must be processed lawfully, fairly and transparently",
        "keywords": ["personal data", "processing", "consent", "lawful"]
    },
    "Section 17": {
        "title": "Data Subject Rights",
        "text": "Right to access, rectification, erasure of personal data",
        "keywords": ["data subject", "access", "rectification", "erasure", "rights"]
    }
}

# Automatic Legal Recommendations based on detected issues
LEGAL_RECOMMENDATIONS = {
    "civil_code": {
        "1108": {
            "missing": [
                "✅ **Ensure all 4 elements present:** Verify consent, capacity, determinate object, and lawful cause",
                "✅ **Document consent clearly:** Use written agreements with signatures",
                "✅ **Verify capacity:** Check that all parties are legally able to contract (age 18+, sound mind)",
                "✅ **Define object precisely:** Specify what is being contracted (goods, services, property)",
                "✅ **Establish lawful cause:** Document the reason/purpose for the contract"
            ],
            "detected": [
                "✓ Contract appears to address essential validity requirements",
                "⚠️ **Review recommendation:** Have a lawyer verify all 4 elements are properly documented"
            ]
        },
        "1131": {
            "missing": [
                "✅ **Add cause/consideration clause:** Clearly state the reason for the contract",
                "✅ **Avoid false causes:** Ensure stated purpose is genuine and achievable",
                "✅ **Check lawfulness:** Verify contract purpose doesn't violate any laws"
            ],
            "detected": [
                "⚠️ **URGENT:** Cause-related issues detected - may invalidate contract",
                "✅ **Immediate action:** Consult lawyer to verify cause is lawful and genuine",
                "✅ **Document rationale:** Add clear statement of contract purpose/reason"
            ]
        },
        "1134": {
            "detected": [
                "✓ Contract has binding legal force once validly formed",
                "⚠️ **Note:** Cannot easily escape obligations - ensure terms are acceptable before signing",
                "✅ **Add termination clause:** Include conditions for lawful contract exit"
            ]
        },
        "1147": {
            "detected": [
                "⚠️ **Breach consequences:** Non-performance may result in damages liability",
                "✅ **Add liquidated damages clause:** Specify predetermined damage amounts",
                "✅ **Include force majeure:** Protect against liability for unforeseeable events",
                "✅ **Define breach clearly:** Specify what constitutes breach and remedies"
            ]
        },
        "1174": {
            "detected": [
                "🚨 **CRITICAL:** Potestative condition detected - may void contract!",
                "✅ **URGENT FIX:** Remove purely discretionary clauses",
                "✅ **Reword conditions:** Make obligations objective, not at one party's whim",
                "✅ **Examples to avoid:** 'at sole discretion', 'if we choose', 'whenever we want'",
                "✅ **Better alternatives:** 'if performance impossible', 'upon 30 days notice', 'for material breach'"
            ]
        },
        "1184": {
            "detected": [
                "✓ Automatic termination right exists for breach",
                "✅ **Clarify termination process:** Add explicit termination clause",
                "✅ **Specify notice requirements:** How much notice for termination?",
                "✅ **Define material breach:** What level of breach triggers termination?",
                "✅ **Add cure period:** Allow time to fix breach before termination"
            ]
        },
        "2262": {
            "detected": [
                "⚠️ **30-year limitation period applies**",
                "✅ **Document preservation:** Keep contract records for 30 years",
                "✅ **Check for shorter periods:** Some claims have 3-year limitation (Article 2277)"
            ]
        },
        "2277": {
            "detected": [
                "🚨 **CRITICAL: 3-YEAR TIME BAR!**",
                "✅ **Urgent action for payment claims:** Must file within 3 years",
                "✅ **Track deadlines:** Set reminders for 2.5 years from breach",
                "✅ **Preserve evidence:** Collect all payment records immediately",
                "✅ **Send demand letters:** Interrupt prescription with written claims",
                "⚠️ **Applies to:** Wages, salaries, professional fees, payments for work"
            ]
        }
    },
    "workers_rights": {
        "Section 27": {
            "missing": [
                "✅ **Create written employment contract:** Required within 14 days of hiring",
                "✅ **Include mandatory terms:** Job title, duties, salary, hours, leave, notice period",
                "✅ **Get employee signature:** Provide copy to employee",
                "✅ **Store securely:** Keep for duration of employment + 3 years"
            ],
            "detected": [
                "✓ Written particulars requirement appears to be addressed",
                "⚠️ **Verify compliance:** Ensure provided within 14 days of employment start",
                "✅ **Check completeness:** Must include all statutory requirements"
            ]
        },
        "Section 36": {
            "detected": [
                "✓ Termination notice requirements apply",
                "✅ **Specify notice periods:** Clearly state notice required (usually 1-3 months)",
                "✅ **Include payment in lieu option:** Allow payment instead of working notice",
                "✅ **Define termination process:** Written notice, exit interview, handover",
                "⚠️ **Notice periods vary by length of service:** Check statutory minimums"
            ]
        },
        "Section 91": {
            "detected": [
                "✓ Severance allowance provisions apply",
                "✅ **Calculate severance:** Typically 15 days per year of service",
                "✅ **Budget for severance:** Include in termination cost planning",
                "✅ **Specify calculation method:** Add severance formula to contract",
                "⚠️ **Mandatory payment:** Cannot be waived or reduced"
            ]
        }
    },
    "data_protection": {
        "Section 5": {
            "missing": [
                "✅ **Add data processing clause:** Specify how personal data will be used",
                "✅ **Obtain explicit consent:** Get written permission for data collection",
                "✅ **Define data purposes:** State clearly why data is collected",
                "✅ **Implement security measures:** Protect personal data from breaches"
            ],
            "detected": [
                "✓ Data processing provisions detected",
                "⚠️ **Ensure GDPR compliance:** Data must be processed lawfully and transparently",
                "✅ **Document consent:** Keep records of data subject consent",
                "✅ **Add privacy policy:** Explain data handling to users",
                "✅ **Appoint data controller:** Designate responsible person"
            ]
        },
        "Section 17": {
            "detected": [
                "✓ Data subject rights provisions apply",
                "✅ **Enable data access:** Provide mechanism for users to access their data",
                "✅ **Allow data correction:** Users can request rectification",
                "✅ **Implement deletion process:** Right to be forgotten/erasure",
                "✅ **Create rights request form:** Make it easy for users to exercise rights",
                "⚠️ **Response deadline:** Must respond within 30 days"
            ]
        }
    }
}

# Risk-based general recommendations
GENERAL_RECOMMENDATIONS = {
    "CRITICAL": [
        "🚨 **URGENT LEGAL REVIEW REQUIRED**",
        "✅ Consult qualified Mauritius lawyer immediately",
        "✅ Do NOT sign contract until issues resolved",
        "✅ Consider contract redrafting",
        "⚠️ High risk of contract being void or unenforceable"
    ],
    "HIGH": [
        "⚠️ **Legal review strongly recommended**",
        "✅ Consult lawyer before finalizing contract",
        "✅ Consider adding protective clauses",
        "✅ Review liability and termination provisions",
        "⚠️ Significant legal risks identified"
    ],
    "MEDIUM": [
        "💡 **Legal review advisable**",
        "✅ Consider consulting lawyer for complex clauses",
        "✅ Ensure all terms are clearly understood",
        "✅ Review and negotiate unfavorable terms"
    ],
    "LOW": [
        "✓ Contract appears relatively standard",
        "💡 Still advisable to have lawyer review",
        "✅ Ensure you understand all obligations"
    ]
}

# Text Extraction
def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    except Exception as e:
        st.error(f"❌ PDF extraction failed: {e}")
        return ""

def extract_text_from_docx(file_bytes: bytes) -> str:
    try:
        doc = docx.Document(io.BytesIO(file_bytes))
        return "\n".join([p.text for p in doc.paragraphs if p.text and not p.text.isspace()])
    except Exception as e:
        st.error(f"❌ DOCX extraction failed: {e}")
        return ""

def extract_text_from_txt(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode(errors="ignore")
    except Exception as e:
        st.error(f"❌ TXT extraction failed: {e}")
        return ""

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\b\d+\s*\n', '', text)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and len(ln.strip()) > 3]
    return "\n".join(lines)

def chunk_text(text: str, approx_tokens: int = CHUNK_TOKENS, overlap: int = 50) -> List[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(len(words), start + approx_tokens)
        chunks.append(" ".join(words[start:end]))
        start = end - overlap
        if start >= len(words) - overlap:
            break
    return chunks

#-----------------------------
# Load NER Model - FIXED VERSION
@st.cache_resource(show_spinner=False)
def load_spacy_model():
    """Load spaCy model with automatic download if not present"""
    try:
        from spacy.cli import download
        
        model_name = "en_core_web_sm"
        
        try:
            # Try to load the model
            nlp = spacy.load(model_name)
        except OSError:
            # Model not found - download it
            with st.spinner(f"📥 Downloading {model_name} (one-time, ~12MB)..."):
                try:
                    download(model_name)
                    nlp = spacy.load(model_name)
                except Exception as e:
                    st.error(f"❌ Failed to download spaCy model: {e}")
                    return None
        
        # Add entity ruler patterns
        if nlp is not None:
            try:
                if "entity_ruler" not in nlp.pipe_names:
                    ruler = nlp.add_pipe("entity_ruler", before="ner", config={"overwrite_ents": True})
                    
                    patterns = [
                        {"label": "LEGAL_TERM", "pattern": [{"LOWER": "liability"}]},
                        {"label": "LEGAL_TERM", "pattern": [{"LOWER": "liabilities"}]},
                        {"label": "LEGAL_TERM", "pattern": [{"TEXT": {"REGEX": "(?i)indemnif.*"}}]},
                        {"label": "LEGAL_TERM", "pattern": [{"LOWER": "hold"}, {"LOWER": "harmless"}]},
                        {"label": "LEGAL_TERM", "pattern": [{"LOWER": "force"}, {"LOWER": "majeure"}]},
                        {"label": "LEGAL_TERM", "pattern": [{"LOWER": "severance"}, {"LOWER": "allowance"}]},
                        {"label": "LEGAL_TERM", "pattern": [{"TEXT": {"REGEX": "(?i)waiv.*"}}]},
                    ]
                    
                    ruler.add_patterns(patterns)
            except Exception as e:
                st.warning(f"⚠️ Could not add entity ruler: {e}")
        
        return nlp
        
    except Exception as e:
        st.error(f"❌ SpaCy loading failed: {e}")
        return None

#-----------------------------------------------
# Contract Analysis Functions

# Known Mauritius locations to filter
MAURITIUS_LOCATIONS = {
    "mauritius", "port louis", "ebene", "quatre bornes", "curepipe", 
    "rose hill", "beau bassin", "vacoas", "phoenix", "moka",
    "flic en flac", "grand baie", "pamplemousses", "triolet",
    "goodlands", "centre de flacq", "mahebourg", "coromandel",
    "floreal", "tamarin", "black river", "souillac", "chemin grenier",
    "republic of mauritius", "port-louis", "plaines wilhems",
    "riviere du rempart", "flacq", "grand port", "savanne",
    "port louis district", "ebene cybercity", "cybercity",
    "bank street", "5th floor", "ground floor", "1st floor", "2nd floor",
    "3rd floor", "4th floor", "6th floor", "7th floor", "8th floor",
    "newton tower", "caudan waterfront", "caudan", "cybertower"
}

# Words that should NEVER be organizations
EXCLUDED_ORG_WORDS = {
    "employer", "employee", "tenant", "landlord", "borrower", "lender",
    "guarantor", "client", "service provider", "developer", "partner",
    "the employer", "the employee", "the tenant", "the landlord",
    "the borrower", "the lender", "the guarantor", "the client",
    "the service provider", "the developer", "the partner",
    "the disclosing party", "the receiving party", "disclosing party", 
    "receiving party", "party a", "party b", "party c",
    "the parties", "parties", "party",
    "agreement", "contract", "the agreement", "the contract",
    "confidential information", "confidentiality", "information",
    "court", "courts", "the courts", "the courts of mauritius",
    "courts of mauritius", "government", "the government",
    "the government of mauritius", "government of mauritius",
    "human resources", "managing director", "chief executive officer",
    "director", "manager", "ceo", "cto", "cfo", "coo",
    "chief operating officer", "chief technology officer",
    "overtime", "maternity", "paternity", "maternity/paternity",
    "confidentiality", "termination", "remuneration", "indemnification",
    "working hours", "annual leave", "sick leave", "probation",
    "no license", "license", "liability", "warranties",
    "the", "a", "an", "and", "or", "of", "to", "for",
    "nothing", "section", "article", "clause"
}

# Words that should NEVER be persons
EXCLUDED_PERSON_WORDS = {
    "human resources", "managing director", "chief executive officer",
    "director", "manager", "ceo", "cto", "cfo", "coo", "employer",
    "employee", "tenant", "landlord", "borrower", "lender", "guarantor",
    "chief operating officer", "chief technology officer",
    "date", "the date", "effective date", "commencement date",
    "witness", "notary", "notary public","the guarantor"
}

def extract_entities(text: str, nlp) -> Dict[str, List[str]]:
    """Extract dates, money, organizations, and people using spaCy NER with filtering"""
    
    if nlp is None:
        return {"dates": [], "money": [], "orgs": [], "persons": []}
    
    try:
        # Truncate text if it exceeds the maximum allowed length
        if len(text) > MAX_NLP_CHARS:
            st.info(f"ℹ️ Text truncated to {MAX_NLP_CHARS:,} chars for entity extraction")
            text = text[:MAX_NLP_CHARS]
        
        doc = nlp(text)

        entities = {
            "dates": [],
            "money": [],
            "orgs": [],
            "persons": []
        }

        for ent in doc.ents:
            ent_text = ent.text.strip()
            ent_lower = ent_text.lower()
            
            # Basic filters
            if len(ent_text) < 3:
                continue
            
            if ent_text.isupper() and len(ent_text) > 4:
                continue
            
            if ent_text.count('(') != ent_text.count(')'):
                continue
            
            if ent_text.startswith(')') or ent_text.endswith('('):
                continue
            if ent_text.startswith('.') or ent_text.startswith(','):
                continue
                
            if any(ent_lower.startswith(p) for p in ['section ', 'article ', 'clause ', 'no ']):
                continue

            # DATE
            if ent.label_ == "DATE":
                entities["dates"].append(ent_text)
            
            # MONEY
            elif ent.label_ == "MONEY":
                entities["money"].append(ent_text)
            
            # ORGANIZATION
            elif ent.label_ == "ORG":
                if ent_lower in MAURITIUS_LOCATIONS:
                    continue
                if ent_lower in EXCLUDED_ORG_WORDS:
                    continue
                if "registered" in ent_lower or "registration" in ent_lower:
                    continue
                if len(ent_text) > 2 and ent_text[-1].isdigit() and '.' in ent_text:
                    continue
                if ent_lower in {"floor", "street", "road", "avenue", "tower", "building"}:
                    continue
                
                is_company = any(x in ent_lower for x in ["ltd", "co.", "inc", "llc", "limited", "corp"])
                is_proper_name = ent_text[0].isupper() and " " not in ent_text and len(ent_text) > 3
                is_multi_word_name = " " in ent_text and ent_text.split()[0][0].isupper()
                
                if is_company or is_proper_name or is_multi_word_name:
                    if ent_lower not in EXCLUDED_ORG_WORDS:
                        entities["orgs"].append(ent_text)
            
            # PERSON
            elif ent.label_ == "PERSON":
                if ent_lower in MAURITIUS_LOCATIONS:
                    continue
                if ent_lower in EXCLUDED_PERSON_WORDS:
                    continue
                if ent_lower in {"date", "agreement", "contract", "witness", "party"}:
                    continue
                if not ent_text[0].isupper():
                    continue
                if ent_lower in {"mr", "mrs", "ms", "dr", "mr.", "mrs.", "ms.", "dr."}:
                    continue
                
                entities["persons"].append(ent_text)

        # Remove duplicates while preserving order
        for key in entities:
            seen = set()
            unique = []
            for item in entities[key]:
                item_lower = item.lower()
                if item_lower not in seen:
                    seen.add(item_lower)
                    unique.append(item)
            entities[key] = unique

        return entities
        
    except Exception as e:
        st.error(f"❌ Entity extraction error: {e}")
        return {"dates": [], "money": [], "orgs": [], "persons": []}

def detect_risk_level(text: str) -> Tuple[str, List[str], float]:
    """Detect risk level and matched keywords in contract text"""
    text_lower = text.lower()
    matched_keywords = []
    risk_score = 0.0
    
    # Check for critical risks
    for keyword in RISK_KEYWORDS["critical"]:
        if keyword in text_lower:
            matched_keywords.append((keyword, "critical"))
            risk_score += 3.0
    
    # Check for high risks
    for keyword in RISK_KEYWORDS["high"]:
        if keyword in text_lower:
            matched_keywords.append((keyword, "high"))
            risk_score += 2.0
    
    # Check for medium risks
    for keyword in RISK_KEYWORDS["medium"]:
        if keyword in text_lower:
            matched_keywords.append((keyword, "medium"))
            risk_score += 1.0
    
    # Determine overall risk level
    if risk_score >= 10:
        risk_level = "CRITICAL"
    elif risk_score >= 5:
        risk_level = "HIGH"
    elif risk_score >= 2:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"
    
    return risk_level, matched_keywords, risk_score

def check_mauritius_compliance(text: str) -> Dict[str, List[str]]:
    """Check compliance with Mauritius regulations"""
    text_lower = text.lower()
    compliance_findings = {}
    
    for law, patterns in MAURITIUS_COMPLIANCE.items():
        matches = [p for p in patterns if p in text_lower]
        if matches:
            compliance_findings[law] = matches
    
    return compliance_findings

def check_law_articles(text: str) -> Dict[str, List[Dict]]:
    """Check which specific law articles are referenced or applicable"""
    text_lower = text.lower()
    findings = {
        "civil_code": [],
        "workers_rights": [],
        "data_protection": []
    }
    
    # Check Civil Code Articles
    for article_num, article_info in CIVIL_CODE_ARTICLES.items():
        # Check if article number mentioned directly
        if f"article {article_num}" in text_lower or f"art. {article_num}" in text_lower:
            findings["civil_code"].append({
                "article": article_num,
                "title": article_info["title"],
                "text": article_info["text"],
                "match_type": "Direct Citation"
            })
        else:
            # Check keywords
            keyword_matches = [kw for kw in article_info["keywords"] if kw in text_lower]
            if len(keyword_matches) >= 2:
                findings["civil_code"].append({
                    "article": article_num,
                    "title": article_info["title"],
                    "text": article_info["text"],
                    "match_type": f"Relevant (keywords: {', '.join(keyword_matches[:3])})"
                })
    
    # Check Workers' Rights Act
    for section, section_info in WORKERS_RIGHTS_PROVISIONS.items():
        if any(kw in text_lower for kw in section_info["keywords"]):
            findings["workers_rights"].append({
                "section": section,
                "title": section_info["title"],
                "text": section_info["text"]
            })

    # Check Data Protection Act
    for section, section_info in DATA_PROTECTION_ARTICLES.items():
        if any(kw in text_lower for kw in section_info["keywords"]):
            findings["data_protection"].append({
                "section": section,
                "title": section_info["title"],
                "text": section_info["text"]
            })
    
    return findings

def generate_recommendations(
    text: str,
    law_articles: Dict[str, List[Dict]],
    risk_level: str
) -> Dict[str, List[str]]:
    """Generate actionable legal recommendations"""
    recommendations = {
        "critical": [],
        "important": [],
        "advisable": [],
        "general": []
    }
    
    # Add general recommendations
    recommendations["general"] = GENERAL_RECOMMENDATIONS.get(risk_level, [])
    
    # Civil Code recommendations
    for article in law_articles.get("civil_code", []):
        article_num = article["article"]
        if article_num in LEGAL_RECOMMENDATIONS["civil_code"]:
            article_recs = LEGAL_RECOMMENDATIONS["civil_code"][article_num].get("detected", [])
            
            for rec in article_recs:
                if "🚨" in rec or "URGENT" in rec or "CRITICAL" in rec:
                    recommendations["critical"].append(rec)
                elif "⚠️" in rec:
                    recommendations["important"].append(rec)
                else:
                    recommendations["advisable"].append(rec)
    
    # Check for missing critical articles
    text_lower = text.lower()
    if not any(kw in text_lower for kw in ["consent", "capacity", "object", "cause"]):
        recommendations["critical"].extend(
            LEGAL_RECOMMENDATIONS["civil_code"]["1108"]["missing"]
        )
    
    # Workers' Rights recommendations
    for provision in law_articles.get("workers_rights", []):
        section = provision["section"]
        if section in LEGAL_RECOMMENDATIONS["workers_rights"]:
            section_recs = LEGAL_RECOMMENDATIONS["workers_rights"][section].get("detected", [])
            for rec in section_recs:
                if "⚠️" in rec:
                    recommendations["important"].append(rec)
                else:
                    recommendations["advisable"].append(rec)
    
    # Data Protection recommendations
    for provision in law_articles.get("data_protection", []):
        section = provision["section"]
        if section in LEGAL_RECOMMENDATIONS["data_protection"]:
            section_recs = LEGAL_RECOMMENDATIONS["data_protection"][section].get("detected", [])
            for rec in section_recs:
                if "⚠️" in rec:
                    recommendations["important"].append(rec)
                else:
                    recommendations["advisable"].append(rec)
    
    # Remove duplicates
    for key in recommendations:
        recommendations[key] = list(dict.fromkeys(recommendations[key]))
    
    return recommendations

def extract_obligations(text: str, nlp) -> List[Dict]:
    """Extract key obligations, deadlines, and payment terms"""
    obligations = []
    
    obligation_keywords = ["shall", "must", "will", "agree to", "required to", "obligated"]
    deadline_keywords = ["within", "by", "before", "after", "days", "date"]
    payment_keywords = ["payment", "pay", "fee", "amount", "price", "cost"]
    
    sentences = text.split('.')
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        
        if any(kw in sentence_lower for kw in obligation_keywords):
            obligation_type = "General"
            
            if any(kw in sentence_lower for kw in deadline_keywords):
                obligation_type = "Deadline"
            elif any(kw in sentence_lower for kw in payment_keywords):
                obligation_type = "Payment"
            
            obligations.append({
                "text": sentence.strip(),
                "type": obligation_type
            })
    
    return obligations[:MAX_OBLIGATIONS_DISPLAY]

def calculate_deadline_warnings(dates: List[str]) -> List[Dict]:
    """Calculate time-bar warnings (3-year Mauritius Civil Code limitation)"""
    warnings = []
    
    for date_str in dates:
        year_match = re.search(r'\b(20\d{2})\b', date_str)
        if year_match:
            year = int(year_match.group(1))
            current_year = datetime.now().year
            years_diff = current_year - year
            
            if years_diff >= 2:
                warnings.append({
                    "date": date_str,
                    "warning": f"⚠️ {years_diff} years old - approaching 3-year Civil Code limitation",
                    "critical": years_diff >= 3
                })
    
    return warnings

def risk_color(level):
    """Return color code for risk level"""
    return {
        "CRITICAL": "#003B6F",
        "HIGH": "#1891C3",
        "MEDIUM": "#3DC6C3",
        "LOW": "#3AC0DA"
    }.get(level, "#50E3C2")

# Embeddings and FAISS management
def build_faiss_index(embedding_dim: int) -> faiss.IndexFlatIP:
    index = faiss.IndexFlatIP(embedding_dim)
    return index

def normalize_vectors(vectors):
    norms = (vectors**2).sum(axis=1, keepdims=True) ** 0.5
    norms[norms == 0] = 1.0
    return vectors / norms

def save_metadata(metadata: List[Tuple[str, int, str]], path: str = METADATA_FILE):
    with open(path, "w", encoding="utf-8") as f:
        for doc_id, idx, text in metadata:
            safe = text.replace("\n", "\\n")
            f.write(f"{doc_id}\t{idx}\t{safe}\n")

def load_metadata(path: str = METADATA_FILE) -> List[Tuple[str, int, str]]:
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t", 2)
            if len(parts) == 3:
                doc_id, idx_s, text_safe = parts
                text = text_safe.replace("\\n", "\n")
                rows.append((doc_id, int(idx_s), text))
    return rows

def calculate_precision_recall(
    retrieved_docs: List[str], 
    relevant_doc: str, 
    embedder, 
    threshold: float = 0.5,
    debug: bool = False
):
    """Calculate precision and recall using cosine similarity threshold"""
    
    if not retrieved_docs or not relevant_doc.strip():
        return 0.0, 0.0, []
    
    try:
        rel_vec = embedder.encode([relevant_doc], convert_to_numpy=True)
    except Exception as e:
        if debug:
            st.error(f"❌ Error encoding relevant doc: {e}")
        return 0.0, 0.0, []

    matches = 0
    similarities = []
    valid_docs = 0
    
    for doc in retrieved_docs:
        if not doc.strip():
            continue
        
        valid_docs += 1
        
        try:
            doc_vec = embedder.encode([doc], convert_to_numpy=True)
            sim = float(cosine_similarity(rel_vec, doc_vec)[0][0])
            similarities.append(sim)
            
            if sim >= threshold:
                matches += 1
                
        except Exception as e:
            if debug:
                st.warning(f"⚠️ Error comparing doc: {e}")
            continue
    
    if debug and similarities:
        st.write(f"📊 Similarities: {[f'{s:.3f}' for s in similarities]}")
        st.write(f"✓ Matches above {threshold}: {matches}/{valid_docs}")
        st.write(f"📈 Max similarity: {max(similarities):.3f}")
        st.write(f"📊 Avg similarity: {np.mean(similarities):.3f}")
    
    if valid_docs == 0:
        return 0.0, 0.0, similarities
    
    precision = matches / valid_docs
    recall = 1.0 if matches > 0 else 0.0
    
    return precision, recall, similarities

def calculate_f1(precision: float, recall: float) -> float:
    """Calculate F1 score"""
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def calculate_mrr(
    results: List[Dict], 
    relevant_doc: str, 
    embedder, 
    threshold: float = 0.5,
    debug: bool = False
) -> float:
    """Calculate MRR based on similarity rank"""
    
    if not results or not relevant_doc.strip():
        return 0.0
    
    try:
        rel_vec = embedder.encode([relevant_doc], convert_to_numpy=True)
    except Exception as e:
        if debug:
            st.error(f"Error encoding for MRR: {e}")
        return 0.0

    for rank, result in enumerate(results, start=1):
        if 'text' not in result or not result['text'].strip():
            continue
        
        try:
            doc_vec = embedder.encode([result['text']], convert_to_numpy=True)
            sim = float(cosine_similarity(rel_vec, doc_vec)[0][0])
            
            if debug:
                st.write(f"Rank {rank}: Similarity = {sim:.4f}")
            
            if sim >= threshold:
                mrr = 1.0 / rank
                if debug:
                    st.success(f"✓ First match at rank {rank}, MRR = {mrr:.4f}")
                return mrr
                
        except Exception as e:
            if debug:
                st.warning(f"Error at rank {rank}: {e}")
            continue
    
    if debug:
        st.warning(f"⚠️ No results above threshold {threshold}")
    return 0.0

def calculate_map(
    results: List[Dict],
    relevant_doc: str,
    embedder,
    threshold: float = 0.5,
    k: int = 5
) -> float:
    """Calculate Mean Average Precision @ K"""
    
    if not results or not relevant_doc:
        return 0.0
    
    try:
        rel_vec = embedder.encode([relevant_doc], convert_to_numpy=True)
    except:
        return 0.0
    
    relevant_count = 0
    precision_sum = 0.0
    
    for rank, result in enumerate(results[:k], start=1):
        if 'text' not in result or not result['text'].strip():
            continue
        
        try:
            doc_vec = embedder.encode([result['text']], convert_to_numpy=True)
            sim = float(cosine_similarity(rel_vec, doc_vec)[0][0])
            
            if sim >= threshold:
                relevant_count += 1
                precision_at_k = relevant_count / rank
                precision_sum += precision_at_k
                
        except:
            continue
    
    if relevant_count == 0:
        return 0.0
    
    return precision_sum / relevant_count

def save_metrics(metrics: Dict, path: str = METRICS_FILE):
    """Save evaluation metrics to JSON"""
    try:
        with open(path, 'w') as f:
            json.dump(metrics, f, indent=2)
    except Exception as e:
        st.warning(f"⚠️ Could not save metrics: {e}")

def load_metrics(path: str = METRICS_FILE) -> Dict:
    """Load evaluation metrics from JSON"""
    try:
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
    except:
        pass
    return {}

#----------------------------------------
# Summarization
def summarize_text(summarizer_model, tokenizer, text: str, max_length: int = 150) -> str:
    """Generate summary of text using T5 model"""
    if summarizer_model is None or tokenizer is None:
        return "Summary generation unavailable in lite mode"
    
    try:
        words = text.split()
        if len(words) > 400:
            chunk_size = 400
            summaries = []
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i:i + chunk_size])
                input_chunk = "summarize: " + chunk
                inputs = tokenizer.encode(input_chunk, return_tensors="pt", truncation=True, max_length=512)
                summary_ids = summarizer_model.generate(inputs, max_length=max_length // 2, min_length=20,
                                                       length_penalty=2.0, num_beams=4, early_stopping=True)
                summaries.append(tokenizer.decode(summary_ids[0], skip_special_tokens=True))
            combined = " ".join(summaries)
            input_text = "summarize: " + combined
        else:
            input_text = "summarize: " + text
        
        inputs = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=512)
        summary_ids = summarizer_model.generate(inputs, max_length=max_length, min_length=30,
                                               length_penalty=2.0, num_beams=4, early_stopping=True)
        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    except Exception as e:
        st.error(f"❌ Summarization failed: {e}")
        return "Summary generation failed"

#------------------------------------------------------
# Models - OPTIMIZED VERSION
@st.cache_resource(show_spinner=False)
def load_models():
    """Load all ML models with proper error handling and progress indicators"""
    
    models = {
        'embedder': None,
        'summarizer': None, 
        'tokenizer': None,
        'nlp': None
    }
    
    progress_container = st.container()
    
    with progress_container:
        st.info("🚀 Loading AI models (first run: 3-5 minutes, cached afterwards)")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # 1. Load Embedder
            status_text.text("📥 Loading embedder model...")
            progress_bar.progress(10)
            
            models['embedder'] = SentenceTransformer(
                EMBEDDING_MODEL_NAME,
                cache_folder=str(CACHE_DIR)
            )
            progress_bar.progress(35)
            status_text.text("✅ Embedder loaded")
            
            # 2. Load Tokenizer
            status_text.text("📥 Loading tokenizer...")
            progress_bar.progress(45)
            
            models['tokenizer'] = T5TokenizerFast.from_pretrained(
                SUMMARIZER_MODEL_NAME,
                cache_dir=str(CACHE_DIR)
            )
            progress_bar.progress(55)
            status_text.text("✅ Tokenizer loaded")
            
            # 3. Load Summarizer
            status_text.text("📥 Loading summarizer model...")
            progress_bar.progress(60)
            
            models['summarizer'] = T5ForConditionalGeneration.from_pretrained(
                SUMMARIZER_MODEL_NAME,
                cache_dir=str(CACHE_DIR)
            )
            progress_bar.progress(85)
            status_text.text("✅ Summarizer loaded")
            
            # 4. Load SpaCy
            status_text.text("📥 Loading spaCy NLP model...")
            progress_bar.progress(90)
            
            models['nlp'] = load_spacy_model()
            progress_bar.progress(100)
            status_text.text("✅ All models loaded successfully!")
            
            # Clean up progress indicators
            import time
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
            progress_container.empty()
            
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            raise e
    
    return models['embedder'], models['summarizer'], models['tokenizer'], models['nlp']

def index_documents(files: List[Tuple[str, bytes]]):
    """Index uploaded documents for search"""
    
    if st.session_state.LITE_MODE:
        st.error("❌ Document indexing unavailable in lite mode")
        return None, None
    
    embedder, _, _, _ = load_models()
    if embedder is None:
        st.error("❌ Embedder not available")
        return None, None
    
    all_chunks = []
    metadata = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        for file_idx, (filename, file_bytes) in enumerate(files):
            status_text.text(f"Processing {filename}...")
            progress_bar.progress((file_idx + 1) / len(files))
            
            ext = filename.lower().split(".")[-1]
            if ext == "pdf":
                raw = extract_text_from_pdf(file_bytes)
            elif ext in ("docx", "doc"):
                raw = extract_text_from_docx(file_bytes)
            else:
                raw = extract_text_from_txt(file_bytes)
            
            if not raw:
                st.warning(f"⚠️ No text extracted from {filename}")
                continue
            
            raw = clean_text(raw)
            chunks = chunk_text(raw)
            
            for i, c in enumerate(chunks):
                all_chunks.append(c)
                metadata.append((filename, i, c))
        
        if not all_chunks:
            st.warning("No text extracted from uploaded files.")
            return None, None
        
        status_text.text("Generating embeddings...")
        vectors = []
        for i in range(0, len(all_chunks), EMBED_BATCH):
            batch = all_chunks[i:i+EMBED_BATCH]
            emb = embedder.encode(batch, convert_to_numpy=True, show_progress_bar=False)
            vectors.append(emb)
        
        vectors = np.vstack(vectors)
        vectors = normalize_vectors(vectors)
        dim = vectors.shape[1]
        index = build_faiss_index(dim)
        index.add(vectors)
        
        faiss.write_index(index, FAISS_INDEX_FILE)
        save_metadata(metadata)
        
        progress_bar.empty()
        status_text.empty()
        st.success(f"✅ Indexed {len(all_chunks)} chunks from {len(files)} contract(s).")
        return index, metadata
        
    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Indexing failed: {e}")
        return None, None

def load_index_and_metadata():
    """Load FAISS index and metadata from disk"""
    if os.path.exists(FAISS_INDEX_FILE):
        index = faiss.read_index(FAISS_INDEX_FILE)
        metadata = load_metadata()
        return index, metadata
    return None, []

def search(query: str, top_k: int = TOP_K):
    """Search indexed documents"""
    try:
        if st.session_state.LITE_MODE:
            st.error("❌ Search unavailable in lite mode")
            return []
        
        embedder, _, _, _ = load_models()
        if embedder is None:
            st.error("❌ Embedder not available")
            return []
        
        index, metadata = load_index_and_metadata()
        
        if index is None or len(metadata) == 0:
            st.warning("No index found. Please upload and index documents first.")
            return []
        
        qv = embedder.encode([query], convert_to_numpy=True)
        qv = normalize_vectors(qv)
        D, I = index.search(qv, top_k)
        
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(metadata):
                continue
            doc_id, chunk_idx, chunk_text = metadata[idx]
            results.append({
                "score": float(score),
                "doc_id": doc_id,
                "chunk_idx": chunk_idx,
                "text": chunk_text
            })
        
        return results
    except Exception as e:
        st.error(f"❌ Search failed: {str(e)}")
        return []

#----------------------------------
# Streamlit UI

# Try to load models
try:
    with st.spinner("🚀 Initializing SmartContract MU..."):
        embedder, summarizer, tokenizer, nlp = load_models()
    st.session_state.LITE_MODE = False
except Exception as e:
    st.error(f"⚠️ Running in LITE MODE: {str(e)}")
    st.info("💡 Some features may be limited. Try refreshing the page. If issue persists, the app may need more memory.")
    embedder, summarizer, tokenizer, nlp = None, None, None, None
    st.session_state.LITE_MODE = True

# Header
st.title("⚖️ SmartContract MU")
st.caption("AI-Powered Legal Contract Analysis for Mauritius")

if st.session_state.LITE_MODE:
    st.warning("⚠️ Running in LITE MODE - Some AI features unavailable")

# Sidebar
with st.sidebar:
    st.markdown("## 📊 SmartContract MU Overview")
    _, metadata = load_index_and_metadata()

    if metadata:
        num_docs = len(set(m[0] for m in metadata))
        num_chunks = len(metadata)

        st.markdown("### 📁 Indexed Contracts")
        col1, col2 = st.columns(2)
        col1.metric("Contracts", num_docs)
        col2.metric("Chunks", num_chunks)

        metrics = load_metrics()
        if metrics:
            st.markdown("### 📈 Evaluation Snapshot")
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    formatted_value = f"{value:.3f}"
                else:
                    formatted_value = str(value)
                
                st.markdown(f"""
                <div style='background-color:#f0f8ff;
                            border-left: 5px solid #1891C3;
                            padding: 8px 12px;
                            border-radius: 6px;
                            margin-bottom: 6px'>
                    <b>{key.upper()}</b>: {formatted_value}
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("📂 No contracts indexed yet")

    st.markdown("---")
    st.markdown("### 🇲🇺 Mauritius Compliance")
    st.markdown("""
    <ul style='padding-left:15px'>
        <li>✓ Workers' Rights Act 2019</li>
        <li>✓ Civil Code Articles</li>
        <li>✓ Data Protection Act 2017</li>
    </ul>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Built with ❤️ for Mauritius legal teams")

# Main Tabs
tabs = st.tabs([
    "📁 Upload Contracts",
    "🔍 Contract Analysis",
    "⚖️ Risk Assessment",
    "🔎 Search & Compare",
    "📈 Evaluation"
])

# TAB 1: Upload & Index
with tabs[0]:
    st.header("📁 Upload and Index Contracts")
    st.markdown("Upload PDF, DOCX, or TXT contracts for analysis")
    
    uploaded = st.file_uploader(
        "Select contract files",
        accept_multiple_files=True,
        type=["pdf", "docx", "txt"],
        key="upload"
    )
    
    if st.button("🚀 Index Contracts", type="primary"):
        if not uploaded:
            st.warning("Please upload at least one contract file")
        elif st.session_state.LITE_MODE:
            st.error("❌ Indexing unavailable in lite mode")
        else:
            files = [(f.name, f.read()) for f in uploaded]
            with st.spinner("Indexing contracts..."):
                index_documents(files)

# TAB 2: Contract Analysis
with tabs[1]:
    st.header("🔍 Complete Contract Analysis")
    
    _, metadata = load_index_and_metadata()
    if not metadata:
        st.info("👆 Please upload and index contracts first")
    else:
        contract_names = sorted(list(set(m[0] for m in metadata)))
        selected_contract = st.selectbox("Select Contract", contract_names)
        
        if st.button("🔬 Analyze Contract", type="primary"):
            contract_chunks = [m[2] for m in metadata if m[0] == selected_contract]
            full_text = "\n\n".join(contract_chunks)
            
            with st.spinner("Analyzing contract..."):
                col1, col2 = st.columns(2)
                
                with col1:
                    # Risk Analysis
                    st.subheader("🚨 Risk Analysis")
                    risk_level, risk_keywords, risk_score = detect_risk_level(full_text)
                    
                    risk_colors_emoji = {
                        "CRITICAL": "🔴",
                        "HIGH": "🟠",
                        "MEDIUM": "🟡",
                        "LOW": "🟢"
                    }
                    
                    st.markdown(f"### {risk_colors_emoji.get(risk_level, '⚪')} {risk_level} RISK")
                    st.metric("Risk Score", f"{risk_score:.1f}")
                    
                    if risk_keywords:
                        st.markdown("**Detected Risk Factors:**")
                        for keyword, level in risk_keywords[:MAX_RISK_KEYWORDS_DISPLAY]:
                            st.markdown(f"- {keyword} ({level})")
                    
                    # Compliance Check
                    st.subheader("✅ Mauritius Compliance")
                    compliance = check_mauritius_compliance(full_text)
                    law_articles = check_law_articles(full_text)
                    recommendations = generate_recommendations(full_text, law_articles, risk_level)
                    
                    if compliance:
                        for law, matches in compliance.items():
                            with st.expander(f"📋 {law.replace('_', ' ').title()}"):
                                for match in matches:
                                    st.markdown(f"✓ {match}")
                    else:
                        st.info("No specific compliance patterns detected")

                    # Civil Code Articles
                    if law_articles["civil_code"]:
                        st.markdown("#### ⚖️ **Relevant Civil Code Articles:**")
                        article_colors = {
                            "1108": "🔴", "1131": "🔴", "1174": "🔴",
                            "1134": "🟠", "1147": "🟠", "1184": "🟠",
                            "2262": "🟡", "2277": "🟠"
                        }
                            
                        for article in law_articles["civil_code"]:
                            emoji = article_colors.get(article['article'], "📜")
                            
                            with st.expander(f"{emoji} Article {article['article']}: {article['title']}"):
                                st.markdown(f"**Match Type:** {article['match_type']}")
                                st.info(article['text'])

                                article_num = article['article']
                                if article_num in LEGAL_RECOMMENDATIONS["civil_code"]:
                                    st.markdown("**📋 Recommendations:**")
                                    for rec in LEGAL_RECOMMENDATIONS["civil_code"][article_num].get("detected", []):
                                        st.markdown(rec)
                                                                
                                if 'explanation' in CIVIL_CODE_ARTICLES[article['article']]:
                                    st.success(f"**💡 Plain English:** {CIVIL_CODE_ARTICLES[article['article']]['explanation']}")
                                
                                if 'risk_level' in CIVIL_CODE_ARTICLES[article['article']]:
                                    risk = CIVIL_CODE_ARTICLES[article['article']]['risk_level']
                                    risk_badge = {
                                        "CRITICAL": "🔴 **CRITICAL**",
                                        "HIGH": "🟠 **HIGH**",
                                        "MEDIUM": "🟡 **MEDIUM**",
                                        "LOW": "🟢 **LOW**"
                                    }
                                    st.markdown(f"**Risk Level:** {risk_badge.get(risk, risk)}")
                                        
                                st.caption(f"*Civil Code of Mauritius - Article {article['article']}*")

                    # Worker's Rights Act
                    if law_articles["workers_rights"]:
                        st.markdown("#### 👷 **Workers' Rights Act 2019:**")
                        for provision in law_articles["workers_rights"]:
                            section_emojis = {"Section 27": "📝", "Section 36": "⏰", "Section 91": "💰"}
                            emoji = section_emojis.get(provision['section'], "📋")
                            
                            with st.expander(f"{emoji} {provision['section']}: {provision['title']}"):
                                st.info(provision['text'])
                                
                                section = provision['section']
                                if section in LEGAL_RECOMMENDATIONS["workers_rights"]:
                                    st.markdown("**📋 Recommendations:**")
                                    for rec in LEGAL_RECOMMENDATIONS["workers_rights"][section].get("detected", []):
                                        st.markdown(rec)
                                
                                st.caption(f"*Workers' Rights Act 2019 - {provision['section']}*")

                    # Data Protection Act
                    if 'data_protection' in law_articles and law_articles["data_protection"]:
                        st.markdown("#### 🔒 **Data Protection Act 2017:**")
                        for provision in law_articles["data_protection"]:
                            with st.expander(f"🔐 {provision['section']}: {provision['title']}"):
                                st.info(provision['text'])
                                
                                section = provision['section']
                                if section in LEGAL_RECOMMENDATIONS["data_protection"]:
                                    st.markdown("**📋 Recommendations:**")
                                    for rec in LEGAL_RECOMMENDATIONS["data_protection"][section].get("detected", []):
                                        st.markdown(rec)
                                
                                if "processing" in provision['title'].lower():
                                    st.warning("⚠️ **GDPR Compliance:** Ensure explicit consent for data processing")
                                elif "rights" in provision['title'].lower():
                                    st.warning("⚠️ **User Rights:** Data subjects have right to access, rectify, and erase their data")
                                
                                st.caption(f"*Data Protection Act 2017 - {provision['section']}*")
                    
                    # Display consolidated recommendations
                    st.markdown("---")
                    st.subheader("💡 Actionable Recommendations")
                    
                    if recommendations["critical"]:
                        with st.expander("🚨 **CRITICAL - Immediate Action Required**", expanded=True):
                            for rec in recommendations["critical"]:
                                st.error(rec)
                    
                    if recommendations["important"]:
                        with st.expander("⚠️ **IMPORTANT - High Priority**", expanded=True):
                            for rec in recommendations["important"]:
                                st.warning(rec)
                    
                    if recommendations["advisable"]:
                        with st.expander("💡 **ADVISABLE - Recommended Actions**"):
                            for rec in recommendations["advisable"]:
                                st.info(rec)
                    
                    if recommendations["general"]:
                        with st.expander("📋 **General Guidance**"):
                            for rec in recommendations["general"]:
                                st.markdown(rec)
                    
                    # Download recommendations
                    if any(recommendations.values()):
                        st.markdown("---")
                        rec_text = "# Legal Recommendations Report\n\n"
                        rec_text += f"**Contract:** {selected_contract}\n"
                        rec_text += f"**Risk Level:** {risk_level}\n"
                        rec_text += f"**Date:** {datetime.now().strftime('%Y-%m-%d')}\n\n"
                        
                        if recommendations["critical"]:
                            rec_text += "## 🚨 CRITICAL - Immediate Action Required\n\n"
                            for rec in recommendations["critical"]:
                                rec_text += f"- {rec}\n"
                            rec_text += "\n"
                        
                        if recommendations["important"]:
                            rec_text += "## ⚠️ IMPORTANT - High Priority\n\n"
                            for rec in recommendations["important"]:
                                rec_text += f"- {rec}\n"
                            rec_text += "\n"
                        
                        if recommendations["advisable"]:
                            rec_text += "## 💡 ADVISABLE - Recommended Actions\n\n"
                            for rec in recommendations["advisable"]:
                                rec_text += f"- {rec}\n"
                            rec_text += "\n"
                        
                        st.download_button(
                            label="📥 Download Recommendations Report",
                            data=rec_text,
                            file_name=f"recommendations_{selected_contract.replace('.', '_')}.md",
                            mime="text/markdown"
                        )

                with col2:
                    # Entity Extraction
                    st.subheader("🏢 Extracted Entities")
                    entities = extract_entities(full_text, nlp)
                    
                    if entities["dates"]:
                        with st.expander(f"📅 Dates ({len(entities['dates'])})"):
                            for date in entities["dates"][:MAX_ENTITIES_DISPLAY]:
                                st.markdown(f"- {date}")
                        
                        warnings = calculate_deadline_warnings(entities["dates"])
                        if warnings:
                            st.warning(f"⚠️ {len(warnings)} potential time-bar concerns")
                            for w in warnings[:5]:
                                st.markdown(f"- {w['date']}: {w['warning']}")
                    
                    if entities["money"]:
                        with st.expander(f"💰 Amounts ({len(entities['money'])})"):
                            for amount in entities["money"][:10]:
                                st.markdown(f"- {amount}")
                    
                    if entities["orgs"]:
                        with st.expander(f"🏢 Organizations ({len(entities['orgs'])})"):
                            for org in entities["orgs"][:10]:
                                st.markdown(f"- {org}")
                    
                    if entities["persons"]:
                        with st.expander(f"👤 Persons ({len(entities['persons'])})"):
                            for person in entities["persons"][:10]:
                                st.markdown(f"- {person}")
                
                # Obligations
                st.subheader("📋 Key Obligations")
                obligations = extract_obligations(full_text, nlp)
                
                if obligations:
                    tab_payment, tab_deadline, tab_general = st.tabs(["💰 Payment", "⏰ Deadlines", "📄 General"])
                    
                    with tab_payment:
                        payment_obs = [o for o in obligations if o["type"] == "Payment"]
                        for ob in payment_obs[:5]:
                            st.markdown(f"• {ob['text']}")
                    
                    with tab_deadline:
                        deadline_obs = [o for o in obligations if o["type"] == "Deadline"]
                        for ob in deadline_obs[:5]:
                            st.markdown(f"• {ob['text']}")
                    
                    with tab_general:
                        general_obs = [o for o in obligations if o["type"] == "General"]
                        for ob in general_obs[:5]:
                            st.markdown(f"• {ob['text']}")
                else:
                    st.info("No explicit obligations detected")
                
                # Summary
                st.subheader("📝 Executive Summary")
                if not st.session_state.LITE_MODE:
                    with st.spinner("Generating summary..."):
                        summary = summarize_text(summarizer, tokenizer, full_text, max_length=200)
                        st.info(summary)
                else:
                    st.warning("⚠️ Summary generation unavailable in lite mode")

# TAB 3: Risk Assessment Dashboard
with tabs[2]:
    st.header("⚖️ Risk Assessment Dashboard")

    _, metadata = load_index_and_metadata()
    if not metadata:
        st.info("👆 Please upload contracts first")
    else:
        st.markdown("### 📋 Contract Risk Overview")

        contract_names = sorted(list(set(m[0] for m in metadata)))
        risk_data = []

        with st.spinner("Analyzing all contracts..."):
            for contract_name in contract_names:
                contract_chunks = [m[2] for m in metadata if m[0] == contract_name]
                full_text = "\n\n".join(contract_chunks)
                risk_level, _, risk_score = detect_risk_level(full_text)
                risk_data.append({
                    "Contract": contract_name,
                    "Risk Level": risk_level,
                    "Risk Score": risk_score
                })

        df = pd.DataFrame(risk_data)
        df.columns = df.columns.str.replace(" ", "_")

        # Render compact cards
        for row in df.itertuples():
            level = row.Risk_Level
            score = row.Risk_Score
            contract = row.Contract
            background = f"{risk_color(level)}33"

            st.markdown(f"""
            <div style='background-color:{background};
                        border-left: 4px solid {risk_color(level)};
                        padding: 6px 10px;
                        border-radius: 6px;
                        margin-bottom: 6px;
                        font-size: 14px'>
                <div style='font-weight:bold; margin-bottom:2px'>{contract}</div>
                <div style='color:#333'><b>Level:</b> {level} | <b>Score:</b> {score:.1f}</div>
            </div>
            """, unsafe_allow_html=True)

        # View toggle
        view = st.radio("📊 Choose view:", ["Summary", "Chart"], horizontal=True)

        risk_counts = df['Risk_Level'].value_counts()
        levels = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]

        if view == "Summary":
            st.markdown("### 📋 Risk Summary")
            for level in levels:
                count = risk_counts.get(level, 0)
                st.markdown(f"""
                <div style='display:inline-block; background-color:{risk_color(level)}33;
                            border-left: 5px solid {risk_color(level)};
                            padding: 10px 20px; margin: 5px;
                            border-radius: 8px; min-width: 120px; text-align:center'>
                    <h4 style='margin:0'>{level}</h4>
                    <b>{count} contract{'s' if count != 1 else ''}</b>
                </div>
                """, unsafe_allow_html=True)

        else:
            st.markdown("### 📊 Risk Score by Contract")
            color_scale = alt.Scale(
                domain=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
                range=["#003B6F", "#1891C3", "#3DC6C3", "#3AC0DA"]
            )

            bar = alt.Chart(df).mark_bar(size=25, opacity=0.85).encode(
                y=alt.Y('Contract', sort='-x', title='Contract'),
                x=alt.X('Risk_Score', title='Risk Score'),
                color=alt.Color('Risk_Level', scale=color_scale, legend=None),
                tooltip=['Contract', 'Risk_Level', 'Risk_Score']
            ).properties(
                height=200,
                title="📊 Risk Score by Contract"
            ).configure_axis(
                labelFontSize=12,
                titleFontSize=13
            ).configure_title(
                fontSize=14,
                anchor='start',
                color='#444'
            )

            st.altair_chart(bar, use_container_width=True)
            st.caption("📊 Each bar shows a contract's risk score and level. Hover to see details.")

# TAB 4: Search & Compare
with tabs[3]:
    st.header("🔍 Search & Compare Contracts")

    _, metadata = load_index_and_metadata()
    if not metadata:
        st.info("👆 Please upload contracts first")
    elif st.session_state.LITE_MODE:
        st.warning("⚠️ Search features unavailable in lite mode")
    else:
        contract_names = sorted(list(set(m[0] for m in metadata)))

        # Search Section
        st.markdown("### 🔎 Search Contracts")
        query = st.text_input("Search for specific clauses or terms")
        col1, col2 = st.columns([3, 1])
        with col1:
            k = st.slider("Number of results", min_value=1, max_value=10, value=5)
        with col2:
            show_summary = st.checkbox("Generate Summary", value=True)

        if st.button("🔍 Search", type="primary"):
            if not query:
                st.warning("Please enter a search query")
            else:
                results = search(query, top_k=k)

                if not results:
                    st.info("No results found")
                else:
                    st.subheader(f"📑 Top {len(results)} Results")

                    concatenated = []
                    for idx, r in enumerate(results, 1):
                        risk_level, _, _ = detect_risk_level(r["text"])
                        with st.expander(
                            f"Result {idx}: {r['doc_id']} (Score: {r['score']:.3f}) - Risk: {risk_level}"
                        ):
                            st.markdown(r["text"])
                            st.caption(f"Chunk {r['chunk_idx']}")
                        concatenated.append(r["text"])

                    if show_summary and concatenated and not st.session_state.LITE_MODE:
                        joined = "\n\n".join(concatenated)
                        if len(joined.split()) >= 20:
                            with st.spinner("Generating summary..."):
                                summary = summarize_text(summarizer, tokenizer, joined)
                                st.subheader("📝 Summary")
                                st.info(summary)

        # Compare Section
        st.markdown("### ⚖️ Compare Two Contracts")
        col1, col2 = st.columns(2)
        with col1:
            contract1 = st.selectbox("Select first contract", contract_names)
        with col2:
            contract2 = st.selectbox("Select second contract", [c for c in contract_names if c != contract1])

        if contract1 and contract2:
            text1 = "\n\n".join([m[2] for m in metadata if m[0] == contract1])
            text2 = "\n\n".join([m[2] for m in metadata if m[0] == contract2])

            risk1, _, score1 = detect_risk_level(text1)
            risk2, _, score2 = detect_risk_level(text2)

            st.markdown("### 📊 Comparison Result")
            st.markdown(f"""
            <div style='display:flex; gap:30px'>
                <div style='background-color:{risk_color(risk1)}33;
                            border-left: 5px solid {risk_color(risk1)};
                            padding: 10px 15px;
                            border-radius: 8px; width:45%'>
                    <h4>{contract1}</h4>
                    <b>Risk Level:</b> {risk1}<br>
                    <b>Risk Score:</b> {score1:.2f}
                </div>
                <div style='background-color:{risk_color(risk2)}33;
                            border-left: 5px solid {risk_color(risk2)};
                            padding: 10px 15px;
                            border-radius: 8px; width:45%'>
                    <h4>{contract2}</h4>
                    <b>Risk Level:</b> {risk2}<br>
                    <b>Risk Score:</b> {score2:.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)

# TAB 5: Evaluation & Metrics
with tabs[4]:
    st.header("📈 Evaluation & Metrics")
    
    if st.session_state.LITE_MODE:
        st.warning("⚠️ Evaluation features unavailable in lite mode")
    else:
        threshold = st.slider("🎯 Similarity Threshold", 0.1, 1.0, 0.5, 0.05,
                              help="0.4-0.6 recommended")
        
        ground_truth = load_ground_truth_cases()
        
        # Add test case
        with st.expander("➕ Add Ground Truth Test Case"):
            with st.form("add_gt_form", clear_on_submit=True):
                test_query = st.text_input("Query", placeholder="e.g., termination notice period")
                relevant_doc = st.text_area("Expected Content (200+ words recommended)", height=120)
                
                submitted = st.form_submit_button("Add Test Case", type="primary")
                
                if submitted:
                    if test_query and relevant_doc:
                        ground_truth[test_query] = relevant_doc
                        save_ground_truth_cases(ground_truth)
                        st.success("✅ Test case added!")
                        st.rerun()
                    else:
                        st.warning("Please fill both fields")
        
        # Show/manage cases (rest of Tab 5 code continues...)
        col1, col2 = st.columns(2)
        with col1:
            show_cases = st.button(f"📋 Show Saved Cases ({len(ground_truth)})" if ground_truth else "📋 No Saved Cases")
        with col2:
            manage_cases = st.button("✏️ Manage Cases") if ground_truth else None
        
        if "show_saved_cases" not in st.session_state:
            st.session_state.show_saved_cases = False
        if "manage_mode" not in st.session_state:
            st.session_state.manage_mode = False
        
        if show_cases:
            st.session_state.show_saved_cases = not st.session_state.show_saved_cases
            st.session_state.manage_mode = False
        
        if manage_cases:
            st.session_state.manage_mode = not st.session_state.manage_mode
            st.session_state.show_saved_cases = False
        
        if st.session_state.show_saved_cases and ground_truth:
            st.markdown("#### 📋 Saved Test Cases")
            for query, doc in ground_truth.items():
                with st.expander(f"🔎 {query}"):
                    st.markdown(f"**Expected Content ({len(doc)} chars):**")
                    st.text(doc[:500] + "..." if len(doc) > 500 else doc)
                    if len(doc) < 100:
                        st.warning("⚠️ Text is short - consider using longer text")
        
        if st.session_state.manage_mode and ground_truth:
            st.markdown("#### ✏️ Manage Test Cases")
            
            selected_query = st.selectbox("Select a test case:", list(ground_truth.keys()), key="select_case")
            
            if selected_query:
                st.markdown(f"**Editing: {selected_query}**")
                
                edited_content = st.text_area(
                    "Edit content:", 
                    value=ground_truth[selected_query], 
                    height=150,
                    key=f"edit_{selected_query}"
                )
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("💾 Save Changes", type="primary"):
                        ground_truth[selected_query] = edited_content
                        save_ground_truth_cases(ground_truth)
                        st.success("✅ Updated!")
                        st.rerun()
                
                with col2:
                    if st.button("🗑️ Delete This Case", type="secondary"):
                        del ground_truth[selected_query]
                        save_ground_truth_cases(ground_truth)
                        st.success("✅ Deleted!")
                        st.rerun()
                
                with col3:
                    if st.button("🗑️ Delete ALL Cases"):
                        save_ground_truth_cases({})
                        st.success("✅ All cases deleted!")
                        st.rerun()
        
        st.divider()
        
        # Quick Test Section
        st.markdown("### 🔬 Quick Test")
        
        if "quick_test_counter" not in st.session_state:
            st.session_state.quick_test_counter = 0
        
        col1, col2 = st.columns([3, 1])
        with col1:
            test_query_input = st.text_input("Test a query:", placeholder="e.g., termination notice period",
                                             key=f"quick_query_{st.session_state.quick_test_counter}")
        with col2:
            test_k = st.number_input("Top K", min_value=1, max_value=10, value=5)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            test_btn = st.button("🧪 Test Query", type="secondary")
        with col2:
            if st.button("🗑️ Clear"):
                st.session_state.quick_test_counter += 1
                st.rerun()
        
        if test_btn:
            if not test_query_input:
                st.warning("Enter a query to test")
            else:
                results = search(test_query_input, top_k=test_k)
                
                if not results:
                    st.error("❌ No results found!")
                else:
                    st.success(f"✓ Found {len(results)} results")
                    
                    for i, r in enumerate(results, 1):
                        with st.expander(f"Result {i}: {r['doc_id']} (Score: {r['score']:.3f})"):
                            st.text(r['text'][:500] + "..." if len(r['text']) > 500 else r['text'])
                    
                    if test_query_input in ground_truth:
                        st.markdown("---")
                        relevant_doc = ground_truth[test_query_input]
                        retrieved_docs = [r['text'] for r in results]
                        
                        precision, recall, _ = calculate_precision_recall(retrieved_docs, relevant_doc, embedder, threshold)
                        f1 = calculate_f1(precision, recall)
                        mrr = calculate_mrr(results, relevant_doc, embedder, threshold)
                        
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Precision", f"{precision:.3f}")
                        c2.metric("Recall", f"{recall:.3f}")
                        c3.metric("F1", f"{f1:.3f}")
                        c4.metric("MRR", f"{mrr:.3f}")
