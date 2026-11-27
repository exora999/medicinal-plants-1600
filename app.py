
import json
import urllib.request
import urllib.parse
import html
import base64
import io as _io
import time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import streamlit as st
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

st.set_page_config(
    page_title="Medicinal Plant Classifier — 1,600 Species",
    page_icon="🌿",
    layout="wide"
)

LANGUAGES = {
    "🇲🇦 العربية": "ar",
    "🇫🇷 Français": "fr",
    "🇬🇧 English": "en"
}

TEXT = {
    "en": {
        "hero": "Medicinal Plant Classifier",
        "subtitle": "Dual-Model Ensemble AI (MobileNetV2 + EfficientNetV2) & Moroccan ethnobotany — <b>1,600 Species</b>",
        "badge_info": "🔬 1,600 Species &bull; Weighted Ensemble AI &bull; Grad-CAM &bull; PFAF &bull; PubChem",
        "input": "📸 Input Plant Image",
        "upload_tab": "📁 Upload File",
        "camera_tab": "📷 Take Photo",
        "upload": "Upload plant image",
        "capture": "Capture plant photo",
        "ready_img": "✅ Image Ready",
        "analyse": "📊  |  Analyse Plant  ➔",
        "upload_first": "Please upload or capture a plant photo first.",
        "running": "🌿 Running Dual-Model Ensemble & generating Grad-CAM...",
        "fetching": "🧪 Retrieving chemical compounds from PubChem...",
        "ood_title": "⚠️ Low Confidence / Out-of-Distribution Image Detected",
        "ood_body": "This image does not clearly match any of the 1,600 trained plant species (it may be a non-plant object, a blurry photo, or an unindexed species). Try uploading a clearer close-up of the leaves or flowers.",
        "ood_override": "👁️ Show Closest Species Match Anyway",
        "top5_title": "🎯 Top 5 Predicted Species (Dual-Model Ensemble)",
        "gradcam_title": "🔍 AI Visual Attention (Grad-CAM Explainability)",
        "gradcam_orig": "Original Input",
        "gradcam_heat": "Model Focus Heatmap (Leaves / Flowers)",
        "gradcam_desc": "Warm regions (red/yellow) highlight the exact botanical structures that drove the neural network's prediction.",
        "ref_gallery": "🖼️ Visual Reference Comparison (iNaturalist)",
        "plant_id": "Plant Identification",
        "scientific": "Scientific Name",
        "common": "Common Name",
        "vernacular": "🇲🇦 Moroccan Vernacular",
        "confidence": "Ensemble Top-1 Confidence",
        "plant_info": "Botanical & Medicinal Profile (PFAF)",
        "family": "🌱 Family",
        "med_rating": "⭐ Medicinal Rating (0–5)",
        "edib_rating": "🍽️ Edibility Rating (0–5)",
        "range": "🗺️ Native Range",
        "properties": "💊 Medicinal Properties",
        "morocco_sec": "🇲🇦 Moroccan Ethnobotanical Context",
        "sahara_lbl": "🏜️ Moroccan Sahara Study",
        "covid_lbl": "🏥 Moroccan National Study (COVID-19)",
        "not_morocco": "Not documented in the integrated Moroccan ethnobotanical studies.",
        "toxic_banner": "🚨 SAFETY & TOXICITY ALERT",
        "safe_banner": "✅ No toxicity alerts recorded in integrated Moroccan or PFAF records.",
        "chemicals": "Chemical Compounds (PubChem)",
        "chem_note": "Chemical compound names are kept in their standard scientific form.",
        "evaluate": "🎯 Evaluate Model Prediction",
        "evaluate_help": "Is the <b>1,600-Species Ensemble's</b> top prediction accurate for this plant?",
        "yes": "✅ Yes, Prediction is Correct",
        "no": "❌ No, Prediction is Wrong",
        "thanks": "Thank you for validating the model's prediction! 🎉",
        "select_true": "Select from Top-5 candidates or type the true scientific name:",
        "custom_sp": "Or enter exact scientific name (e.g., Rosmarinus officinalis):",
        "submit": "📨 Submit Correction",
        "saving": "Saving correction...",
        "saved": "Correction saved! Thank you for helping improve the dataset 🎉",
        "save_error": "Error saving feedback: ",
        "ready": "Ready to Analyse",
        "ready_help": "Upload or capture a plant photo and click <b>Analyse Plant</b>",
        "academic": "⚠️ <b>Academic & Educational Project</b> — Do NOT use this tool for medical diagnosis or consumption decisions.",
        "dataset": "Database: 1,600 Species • 73,948 Images (iNaturalist + PFAF + Moroccan Studies)"
    },
    "fr": {
        "hero": "Classificateur de Plantes Médicinales",
        "subtitle": "Ensemble IA Double Modèle (MobileNetV2 + EfficientNetV2) & ethnobotanique marocaine — <b>1 600 Espèces</b>",
        "badge_info": "🔬 1 600 Espèces &bull; IA Ensemble Pondéré &bull; Grad-CAM &bull; PFAF &bull; PubChem",
        "input": "📸 Image de la plante",
        "upload_tab": "📁 Importer un fichier",
        "camera_tab": "📷 Prendre une photo",
        "upload": "Importer une image de plante",
        "capture": "Photographier une plante",
        "ready_img": "✅ Image prête",
        "analyse": "📊  |  Analyser la plante  ➔",
        "upload_first": "Veuillez importer ou prendre une photo d’une plante.",
        "running": "🌿 Analyse par ensemble double modèle et carte Grad-CAM...",
        "fetching": "🧪 Récupération des composés chimiques (PubChem)...",
        "ood_title": "⚠️ Image hors distribution / Confiance trop faible",
        "ood_body": "Cette image ne correspond clairement à aucune des 1 600 espèces entraînées (objet non végétal, photo floue ou espèce absente de la base). Essayez une photo nette des feuilles ou fleurs.",
        "ood_override": "👁️ Afficher quand même l’espèce la plus proche",
        "top5_title": "🎯 Top 5 des espèces prédites (Ensemble Double Modèle)",
        "gradcam_title": "🔍 Attention visuelle de l’IA (Grad-CAM)",
        "gradcam_orig": "Image d’origine",
        "gradcam_heat": "Zones de décision (Feuilles / Fleurs)",
        "gradcam_desc": "Les zones chaudes (rouge/jaune) montrent les structures botaniques utilisées par le réseau de neurones pour identifier la plante.",
        "ref_gallery": "🖼️ Comparaison visuelle (iNaturalist)",
        "plant_id": "Identification de la plante",
        "scientific": "Nom scientifique",
        "common": "Nom commun",
        "vernacular": "🇲🇦 Nom vernaculaire marocain",
        "confidence": "Confiance Ensemble Top-1",
        "plant_info": "Profil botanique et médicinal (PFAF)",
        "family": "🌱 Famille",
        "med_rating": "⭐ Note médicinale (0–5)",
        "edib_rating": "🍽️ Note comestible (0–5)",
        "range": "🗺️ Répartition naturelle",
        "properties": "💊 Propriétés médicinales",
        "morocco_sec": "🇲🇦 Contexte ethnobotanique marocain",
        "sahara_lbl": "🏜️ Étude du Sahara marocain",
        "covid_lbl": "🏥 Étude nationale marocaine (COVID-19)",
        "not_morocco": "Non répertoriée dans les études ethnobotaniques marocaines intégrées.",
        "toxic_banner": "🚨 ALERTE DE SÉCURITÉ ET DE TOXICITÉ",
        "safe_banner": "✅ Aucune alerte de toxicité enregistrée dans les bases marocaines ou PFAF.",
        "chemicals": "Composés chimiques (PubChem)",
        "chem_note": "Les noms des composés chimiques sont conservés sous leur forme scientifique normalisée.",
        "evaluate": "🎯 Évaluer la prédiction du modèle",
        "evaluate_help": "La prédiction principale de l’<b>Ensemble 1 600 Espèces</b> est-elle correcte ?",
        "yes": "✅ Oui, la prédiction est correcte",
        "no": "❌ Non, la prédiction est incorrecte",
        "thanks": "Merci d’avoir validé la prédiction du modèle ! 🎉",
        "select_true": "Choisissez parmi le Top 5 ou saisissez le nom scientifique exact :",
        "custom_sp": "Ou entrez le nom scientifique (ex. Rosmarinus officinalis) :",
        "submit": "📨 Envoyer la correction",
        "saving": "Enregistrement de la correction...",
        "saved": "Correction enregistrée ! Merci pour votre contribution 🎉",
        "save_error": "Erreur lors de l’enregistrement : ",
        "ready": "Prêt à analyser",
        "ready_help": "Importez une photo de plante puis cliquez sur <b>Analyser la plante</b>",
        "academic": "⚠️ <b>Projet académique et éducatif</b> — N’utilisez PAS cet outil pour un diagnostic médical ou une décision de consommation.",
        "dataset": "Base de données : 1 600 Espèces • 73 948 Images (iNaturalist + PFAF + Études Marocaines)"
    },
    "ar": {
        "hero": "نظام التعرّف على النباتات الطبية",
        "subtitle": "الذكاء الاصطناعي التجميعي المزدوج (MobileNetV2 + EfficientNetV2) والسياق المغربي — <b>1,600 نوع نباتي</b>",
        "badge_info": "🔬 1,600 نوع &bull; دمج نموذجين Ensemble &bull; Grad-CAM &bull; PFAF &bull; PubChem",
        "input": "📸 صورة النبتة",
        "upload_tab": "📁 رفع ملف",
        "camera_tab": "📷 التقاط صورة",
        "upload": "ارفع صورة النبتة",
        "capture": "التقط صورة للنبتة",
        "ready_img": "✅ الصورة جاهزة",
        "analyse": "📊  |  تحليل النبتة  ➔",
        "upload_first": "يرجى رفع صورة النبتة أو التقاطها أولاً.",
        "running": "🌿 جارٍ تحليل النبتة عبر النموذجين المدمجين وتوليد خريطة Grad-CAM...",
        "fetching": "🧪 جارٍ جلب المركبات الكيميائية من PubChem...",
        "ood_title": "⚠️ تنبيه: الصورة خارج النطاق أو غير واضحة",
        "ood_body": "هذه الصورة لا تطابق بشكل موثوق أياً من الأنواع الـ 1,600 المدرَّبة (قد تكون جسماً غير نباتي، أو صورة مشوشة، أو نوعاً غير مدرج في قاعدة البيانات). يُرجى رفع صورة أوضح للأوراق أو الأزهار.",
        "ood_override": "👁️ عرض أقرب تخمين على أي حال",
        "top5_title": "🎯 أكثر 5 أنواع متوقعة (دمج النموذجين Ensemble)",
        "gradcam_title": "🔍 تفسير قرار الذكاء الاصطناعي (خريطة تركيز Grad-CAM)",
        "gradcam_orig": "الصورة الأصلية",
        "gradcam_heat": "مناطق تركيز النموذج (الأوراق / الأزهار)",
        "gradcam_desc": "المناطق الدافئة (الأحمر والأصفر) تُظهر أجزاء النبتة التي اعتمد عليها النموذج لاتخاذ قراره.",
        "ref_gallery": "🖼️ صور مرجعية للمقارنة البصرية (iNaturalist)",
        "plant_id": "بطاقة هوية النبتة",
        "scientific": "الاسم العلمي",
        "common": "الاسم الشائع",
        "vernacular": "🇲🇦 الاسم بالدارجة المغربية",
        "confidence": "نسبة ثقة التعرف المدمجة",
        "plant_info": "الملف النباتي والطبي (PFAF)",
        "family": "🌱 العائلة النباتية",
        "med_rating": "⭐ التقييم الطبي (0–5)",
        "edib_rating": "🍽️ التقييم الغذائي (0–5)",
        "range": "🗺️ الموطن الأصلي",
        "properties": "💊 الخصائص والاستعمالات الطبية",
        "morocco_sec": "🇲🇦 السياق الإثنوبوتاني المغربي",
        "sahara_lbl": "🏜️ دراسة أقاليم الصحراء المغربية",
        "covid_lbl": "🏥 الدراسة الوطنية المغربية (كوفيد-19)",
        "not_morocco": "غير مدرجة ضمن الأنواع الموثقة في الدراستين المغربيتين المدمجتين.",
        "toxic_banner": "🚨 تنبيه السلامة والسمية",
        "safe_banner": "✅ لا توجد تنبيهات سمية مسجلة في السجلات المغربية أو PFAF.",
        "chemicals": "المركبات الكيميائية الفعّالة (PubChem)",
        "chem_note": "تُترك أسماء المركبات الكيميائية بصيغتها العلمية القياسية حفاظاً على دقتها.",
        "evaluate": "🎯 تقييم دقة النموذج",
        "evaluate_help": "هل التوقع الأول للنظام المدمج <b>1,600 نوع</b> صحيح لهذه النبتة؟",
        "yes": "✅ نعم، التوقع صحيح",
        "no": "❌ لا، التوقع غير صحيح",
        "thanks": "شكراً لتأكيد نتيجة النموذج! 🎉",
        "select_true": "اختر النوع الصحيح من القائمة أو اكتب اسمه العلمي:",
        "custom_sp": "أو اكتب الاسم العلمي الصحيح (مثال: Rosmarinus officinalis):",
        "submit": "📨 إرسال التصحيح",
        "saving": "جارٍ حفظ التصحيح...",
        "saved": "تم حفظ التصحيح بنجاح! شكراً لمساهمتك في تحسين قاعدة البيانات 🎉",
        "save_error": "خطأ أثناء حفظ الملاحظة: ",
        "ready": "جاهز للتحليل",
        "ready_help": "ارفع صورة النبتة من الجهة الجانبية ثم اضغط على <b>تحليل النبتة</b>",
        "academic": "⚠️ <b>مشروع أكاديمي وتعليمي</b> — لا تستخدم هذه الأداة للتشخيص الطبي أو لاتخاذ قرارات الاستهلاك.",
        "dataset": "قاعدة البيانات: 1,600 نوع • 73,948 صورة (iNaturalist + PFAF + الدراسات المغربية)"
    }
}

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&family=Cairo:wght@500;600;700;800&display=swap');

  html, body, [class*="css"] {
      font-family: 'Plus Jakarta Sans', 'Cairo', sans-serif;
  }
  header[data-testid="stHeader"] {
      background: transparent !important;
  }
  .block-container {
      padding-top: 2.0rem !important;
  }
  .stApp {
      background: radial-gradient(circle at 50% -15%, #063729 0%, #0b1324 45%, #030712 100%);
      color: #f8fafc;
  }
  .top-lang-badge {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      background: rgba(15, 23, 42, 0.72);
      border: 1px solid rgba(52, 211, 153, 0.32);
      border-radius: 999px;
      padding: 7px 16px;
      font-size: 0.82rem;
      font-weight: 700;
      color: #cbd5e1;
      backdrop-filter: blur(10px);
      margin-top: 4px;
      transition: all 0.25s ease;
  }
  .top-lang-badge:hover {
      border-color: #34d399;
      color: #ffffff;
      transform: translateY(-1px);
  }
  div[data-testid="stSelectbox"] > label {
      display: none !important;
  }
  div[data-testid="stSelectbox"] > div > div {
      background: linear-gradient(135deg, rgba(15, 23, 42, 0.92), rgba(30, 41, 59, 0.88)) !important;
      border: 1px solid rgba(52, 211, 153, 0.35) !important;
      border-radius: 14px !important;
      color: #f8fafc !important;
      font-weight: 700 !important;
      backdrop-filter: blur(12px);
      box-shadow: 0 6px 18px rgba(0, 0, 0, 0.3);
      transition: all 0.25s ease !important;
      cursor: pointer !important;
  }
  div[data-testid="stSelectbox"] > div > div:hover {
      border-color: #4ade80 !important;
      transform: translateY(-2px);
      box-shadow: 0 0 18px rgba(74, 222, 128, 0.28);
  }
  div[data-testid="stSelectbox"] svg {
      fill: #4ade80 !important;
  }
  .hero-box {
      background: linear-gradient(135deg, rgba(6, 78, 59, 0.85), rgba(15, 118, 110, 0.75), rgba(15, 23, 42, 0.92));
      backdrop-filter: blur(14px);
      border: 1px solid rgba(74, 222, 128, 0.32);
      border-radius: 26px;
      padding: 28px 24px;
      text-align: center;
      color: white;
      margin-top: 8px;
      margin-bottom: 26px;
      box-shadow: 0 20px 45px -15px rgba(16, 185, 129, 0.28);
      transition: transform 0.3s ease, border-color 0.3s ease;
  }
  .hero-box:hover {
      border-color: rgba(74, 222, 128, 0.55);
  }
  .hero-badge {
      display: inline-block;
      background: rgba(15, 23, 42, 0.45);
      backdrop-filter: blur(6px);
      border: 1px solid rgba(255, 255, 255, 0.22);
      border-radius: 999px;
      padding: 6px 20px;
      font-size: 0.88rem;
      color: #e2e8f0;
      margin-top: 10px;
  }
  .ood-alert-box {
      background: linear-gradient(135deg, rgba(120, 53, 15, 0.88), rgba(146, 64, 14, 0.72));
      border: 1.5px solid #fbbf24;
      border-radius: 20px;
      padding: 20px 22px;
      margin-bottom: 18px;
      color: #fef3c7;
      box-shadow: 0 14px 32px rgba(217, 119, 6, 0.25);
      transition: transform 0.25s ease, box-shadow 0.25s ease;
  }
  .ood-alert-box:hover {
      transform: translateY(-2px);
      box-shadow: 0 18px 36px rgba(217, 119, 6, 0.35);
  }
  .model-card-primary {
      background: linear-gradient(180deg, rgba(15, 23, 42, 0.88) 0%, rgba(17, 24, 39, 0.95) 100%);
      backdrop-filter: blur(12px);
      border: 1.5px solid rgba(74, 222, 128, 0.45);
      border-radius: 22px;
      padding: 22px 22px;
      box-shadow: 0 12px 32px -10px rgba(0, 0, 0, 0.45);
      margin-bottom: 18px;
      transition: transform 0.25s cubic-bezier(0.22, 1, 0.36, 1), border-color 0.25s ease, box-shadow 0.25s ease;
  }
  .model-card-primary:hover {
      transform: translateY(-3px);
      border-color: #4ade80;
      box-shadow: 0 16px 38px -8px rgba(74, 222, 128, 0.22);
  }
  .bar-wrap {
      margin: 8px 0 10px 0;
      padding: 8px 12px;
      border-radius: 12px;
      background: rgba(30, 41, 59, 0.42);
      border: 1px solid rgba(255, 255, 255, 0.05);
      transition: all 0.22s ease;
  }
  .bar-wrap:hover {
      background: rgba(30, 41, 59, 0.82);
      border-color: rgba(74, 222, 128, 0.35);
      transform: translateX(4px);
  }
  .bar-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 0.95rem;
      font-weight: 700;
      margin-bottom: 6px;
      color: #f8fafc;
  }
  .bar-track {
      background: rgba(2, 6, 23, 0.85);
      border-radius: 999px;
      height: 16px;
      width: 100%;
      overflow: hidden;
      border: 1px solid rgba(255, 255, 255, 0.08);
      padding: 2px;
  }
  .bar-fill-top {
      height: 10px;
      border-radius: 999px;
      background: linear-gradient(90deg, #16a34a, #22c55e, #4ade80);
  }
  .bar-fill-sub {
      height: 10px;
      border-radius: 999px;
      background: linear-gradient(90deg, #0284c7, #0d9488, #2dd4bf);
  }
  .ref-card {
      background: rgba(15, 23, 42, 0.88);
      border: 1px solid rgba(148, 163, 184, 0.25);
      border-radius: 18px;
      padding: 10px;
      text-align: center;
      margin-bottom: 10px;
      transition: transform 0.24s cubic-bezier(0.22, 1, 0.36, 1), border-color 0.24s ease, box-shadow 0.24s ease;
  }
  .ref-card:hover {
      transform: translateY(-4px);
      border-color: #4ade80;
      box-shadow: 0 14px 28px rgba(0, 0, 0, 0.4), 0 0 18px rgba(74, 222, 128, 0.18);
  }
  .ref-card img {
      width: 100%;
      height: 185px;
      object-fit: cover;
      border-radius: 12px;
      margin-bottom: 8px;
  }
  .ref-caption {
      font-size: 0.84rem;
      font-weight: 700;
      color: #a7f3d0;
  }
  .toxic-alert-box {
      background: linear-gradient(135deg, rgba(127, 29, 29, 0.85), rgba(153, 27, 27, 0.65));
      border: 1.5px solid #f87171;
      border-radius: 18px;
      padding: 16px 20px;
      margin: 14px 0;
      color: #fef2f2;
      box-shadow: 0 10px 26px rgba(220, 38, 38, 0.25);
      transition: transform 0.25s ease, box-shadow 0.25s ease;
  }
  .toxic-alert-box:hover {
      transform: translateY(-2px);
      box-shadow: 0 14px 30px rgba(220, 38, 38, 0.38);
  }
  .safe-alert-box {
      background: rgba(15, 23, 42, 0.85);
      border: 1px solid rgba(52, 211, 153, 0.35);
      border-radius: 16px;
      padding: 13px 18px;
      margin: 14px 0;
      color: #a7f3d0;
      font-weight: 600;
      transition: all 0.25s ease;
  }
  .safe-alert-box:hover {
      border-color: #4ade80;
      transform: translateY(-2px);
  }
  .feedback-box {
      background: rgba(15, 23, 42, 0.78);
      border: 1px solid rgba(148, 163, 184, 0.22);
      border-radius: 18px;
      padding: 18px 22px;
      margin-top: 16px;
      transition: border-color 0.25s ease;
  }
  .feedback-box:hover {
      border-color: rgba(74, 222, 128, 0.45);
  }

  .stButton>button {
      border-radius: 14px !important;
      font-weight: 700 !important;
      background: rgba(30, 41, 59, 0.88) !important;
      border: 1px solid rgba(148, 163, 184, 0.25) !important;
      color: #ffffff !important;
      padding: 11px 18px !important;
      transition: transform 0.18s cubic-bezier(0.22, 1, 0.36, 1), box-shadow 0.22s ease, background 0.22s ease, border-color 0.22s ease !important;
      width: 100% !important;
      cursor: pointer !important;
  }
  .stButton>button:hover {
      background: linear-gradient(135deg, #16a34a, #15803d) !important;
      border-color: #4ade80 !important;
      transform: translateY(-3px) !important;
      box-shadow: 0 12px 24px -4px rgba(34, 197, 94, 0.48) !important;
  }
  .stButton>button:active {
      transform: translateY(1px) scale(0.98) !important;
      box-shadow: 0 4px 10px rgba(0, 0, 0, 0.4) !important;
  }

  .st-key-wrong_btn_trigger button:hover {
      background: linear-gradient(135deg, #dc2626, #991b1b) !important;
      border-color: #f87171 !important;
      transform: translateY(-3px) !important;
      box-shadow: 0 12px 24px -4px rgba(220, 38, 38, 0.55) !important;
  }
  .st-key-wrong_btn_trigger button:active {
      transform: translateY(1px) scale(0.98) !important;
  }

  .st-key-analyse_btn button {
      background: linear-gradient(90deg, #16a34a 0%, #22c55e 50%, #10b981 100%) !important;
      border: 1.5px solid #4ade80 !important;
      border-radius: 999px !important;
      color: #ffffff !important;
      font-size: 1.05rem !important;
      font-weight: 800 !important;
      padding: 14px 24px !important;
      box-shadow: 0 8px 25px -4px rgba(34, 197, 94, 0.55) !important;
      letter-spacing: 0.3px !important;
  }
  .st-key-analyse_btn button:hover {
      background: linear-gradient(90deg, #15803d 0%, #16a34a 50%, #059669 100%) !important;
      box-shadow: 0 14px 34px -2px rgba(74, 222, 128, 0.78) !important;
      transform: translateY(-3px) !important;
  }
  .st-key-analyse_btn button:active {
      transform: translateY(2px) scale(0.97) !important;
      box-shadow: 0 4px 14px rgba(34, 197, 94, 0.4) !important;
  }

  [data-testid="stFileUploader"] section {
      background: rgba(15, 23, 42, 0.75) !important;
      border: 1.5px dashed rgba(148, 163, 184, 0.35) !important;
      border-radius: 20px !important;
      padding: 16px !important;
      transition: all 0.25s ease !important;
  }
  [data-testid="stFileUploader"] section:hover {
      border-color: #4ade80 !important;
      background: rgba(30, 41, 59, 0.65) !important;
      transform: translateY(-2px);
  }
  [data-testid="stFileUploader"] section button {
      background: rgba(30, 41, 59, 0.95) !important;
      border: 1px solid rgba(74, 222, 128, 0.4) !important;
      border-radius: 12px !important;
      color: #4ade80 !important;
      font-weight: 700 !important;
      transition: all 0.2s ease !important;
  }
  [data-testid="stFileUploader"] section button:hover {
      background: #16a34a !important;
      color: #ffffff !important;
      transform: translateY(-2px);
  }

  .preview-wrapper {
      position: relative;
      background: #ffffff;
      border-radius: 22px;
      padding: 10px;
      margin-top: 12px;
      margin-bottom: 14px;
      box-shadow: 0 14px 32px -8px rgba(0, 0, 0, 0.6);
      border: 2px solid rgba(74, 222, 128, 0.35);
      overflow: hidden;
      transition: transform 0.25s ease, box-shadow 0.25s ease;
  }
  .preview-wrapper:hover {
      transform: translateY(-3px);
      box-shadow: 0 18px 38px -6px rgba(74, 222, 128, 0.25);
  }
  .preview-wrapper img {
      width: 100%;
      border-radius: 14px;
      display: block;
      object-fit: contain;
      max-height: 290px;
  }
  .ready-pill {
      position: absolute;
      top: 18px;
      right: 18px;
      background: rgba(220, 252, 231, 0.95);
      color: #15803d;
      border: 1px solid #86efac;
      border-radius: 999px;
      padding: 5px 14px;
      font-size: 0.8rem;
      font-weight: 800;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
  }

  .stTabs [data-baseweb="tab-list"] {
      gap: 10px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.1);
      padding-bottom: 6px;
  }
  .stTabs [data-baseweb="tab"] {
      background-color: rgba(30, 41, 59, 0.6);
      border: 1px solid rgba(255, 255, 255, 0.1);
      border-radius: 12px;
      padding: 8px 20px;
      color: #cbd5e1;
      font-weight: 600;
      transition: all 0.2s ease !important;
  }
  .stTabs [data-baseweb="tab"]:hover {
      border-color: rgba(74, 222, 128, 0.5) !important;
      color: #ffffff !important;
      transform: translateY(-2px);
  }
  .stTabs [aria-selected="true"] {
      background: linear-gradient(135deg, rgba(22, 163, 74, 0.35), rgba(15, 23, 42, 0.85)) !important;
      border-color: #4ade80 !important;
      color: #ffffff !important;
  }

  .plant-section-heading {
      display: flex;
      align-items: center;
      gap: 11px;
      margin: 20px 0 12px 0;
      padding: 12px 16px;
      border-radius: 15px;
      background: linear-gradient(100deg, rgba(15, 23, 42, 0.92), rgba(30, 41, 59, 0.78));
      border: 1px solid rgba(148, 163, 184, 0.22);
      box-shadow: inset 0 1px 0 rgba(255,255,255,.05);
      color: #f8fafc;
      font-size: 1.06rem;
      font-weight: 800;
      transition: border-color 0.25s ease, transform 0.25s ease;
  }
  .plant-section-heading:hover {
      border-color: rgba(74, 222, 128, 0.45);
      transform: translateY(-1px);
  }
  .plant-section-heading .section-mark {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 32px;
      height: 32px;
      border-radius: 10px;
      background: rgba(52, 211, 153, 0.14);
      border: 1px solid rgba(52, 211, 153, 0.32);
      color: #6ee7b7;
      flex: 0 0 32px;
  }

  .plant-info-card {
      position: relative;
      min-height: 90px;
      height: calc(100% - 10px);
      background: linear-gradient(145deg, rgba(15, 23, 42, 0.92) 0%, rgba(30, 41, 59, 0.84) 60%, rgba(17, 24, 39, 0.96) 100%);
      border: 1px solid rgba(148, 163, 184, 0.22);
      border-radius: 17px;
      padding: 16px 19px 17px 21px;
      margin: 6px 0 12px 0;
      box-shadow: 0 10px 26px rgba(0, 0, 0, 0.28), inset 0 1px 0 rgba(255, 255, 255, 0.06);
      backdrop-filter: blur(14px);
      overflow: hidden;
      transition: transform 0.24s cubic-bezier(0.22, 1, 0.36, 1), border-color 0.24s ease, box-shadow 0.24s ease, background 0.24s ease;
      cursor: default;
  }
  .plant-info-card::before {
      content: "";
      position: absolute;
      top: 12px;
      bottom: 12px;
      left: 0;
      width: 4px;
      border-radius: 0 5px 5px 0;
      background: linear-gradient(180deg, #4ade80, #0d9488);
      transition: width 0.2s ease, background 0.2s ease;
  }
  .plant-info-card:hover {
      transform: translateY(-4px);
      border-color: rgba(74, 222, 128, 0.65);
      background: linear-gradient(145deg, rgba(20, 35, 55, 0.96) 0%, rgba(15, 65, 55, 0.72) 100%);
      box-shadow: 0 16px 32px rgba(0, 0, 0, 0.38), 0 0 22px rgba(52, 211, 153, 0.16);
  }
  .plant-info-card:hover::before {
      width: 5px;
      background: linear-gradient(180deg, #86efac, #2dd4bf);
  }
  .plant-info-title {
      display: block;
      font-size: .79rem;
      color: #6ee7b7;
      font-weight: 800;
      margin-bottom: 7px;
      letter-spacing: .03em;
  }
  .plant-info-value {
      font-size: .97rem;
      line-height: 1.72;
      color: #f1f5f9;
      overflow-wrap: anywhere;
      white-space: pre-wrap;
  }

  .chemical-panel {
      background: linear-gradient(145deg, rgba(15, 23, 42, 0.9), rgba(30, 41, 59, 0.78));
      border: 1px solid rgba(148, 163, 184, 0.24);
      border-radius: 18px;
      padding: 15px 17px;
      margin: 8px 0 16px 0;
      transition: border-color 0.25s ease, transform 0.25s ease;
  }
  .chemical-panel:hover {
      border-color: rgba(74, 222, 128, 0.5);
      transform: translateY(-2px);
  }
  .chemical-pill {
      display: inline-block;
      direction: ltr;
      background: rgba(30, 41, 59, 0.9);
      border: 1px solid rgba(56, 189, 248, 0.35);
      border-radius: 10px;
      padding: 7px 12px;
      margin: 5px 6px 5px 0;
      color: #bae6fd;
      font-size: .84rem;
      font-weight: 700;
      transition: all 0.2s ease;
  }
  .chemical-pill:hover {
      background: rgba(14, 116, 144, 0.45);
      border-color: #38bdf8;
      color: #ffffff;
      transform: translateY(-2px);
  }

  [dir="rtl"] .plant-info-card {
      padding: 16px 21px 17px 19px;
      text-align: right;
  }
  [dir="rtl"] .plant-info-card::before {
      left: auto;
      right: 0;
      border-radius: 5px 0 0 5px;
  }
  [dir="rtl"] .chemical-pill {
      margin: 5px 0 5px 6px;
  }
  [dir="rtl"] .bar-wrap:hover {
      transform: translateX(-4px);
  }
</style>
""", unsafe_allow_html=True)

top_left, top_right = st.columns([3.1, 1.1])
with top_left:
    st.markdown(
        '<div class="top-lang-badge">🌿 Dual-Model Ensemble AI &bull; 1,600 Species &bull; Grad-CAM &bull; PFAF &bull; PubChem</div>',
        unsafe_allow_html=True
    )
with top_right:
    language_label = st.selectbox(
        "Language",
        list(LANGUAGES.keys()),
        index=0,
        key="site_language",
        label_visibility="collapsed"
    )

LANG = LANGUAGES[language_label]

def tr(key):
    return TEXT[LANG].get(key, TEXT["en"].get(key, key))

def _split_text_chunks(text: str, max_len: int = 1300):
    chunks = []
    remaining = text.strip()
    while remaining:
        if len(remaining) <= max_len:
            chunks.append(remaining)
            break
        cut = max(
            remaining.rfind("\n", 0, max_len),
            remaining.rfind(". ", 0, max_len),
            remaining.rfind("; ", 0, max_len),
            remaining.rfind(", ", 0, max_len),
            remaining.rfind(" ", 0, max_len),
        )
        if cut < 200:
            cut = max_len
        else:
            cut += 1
        chunks.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    return chunks

def _google_translate_chunk(chunk: str, target: str) -> str:
    params = urllib.parse.urlencode({
        "client": "gtx",
        "sl": "auto",
        "tl": target,
        "dt": "t",
        "q": chunk,
    })
    url = f"https://translate.googleapis.com/translate_a/single?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=12) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    parts = [seg[0] for seg in payload[0] if seg and seg[0]]
    return "".join(parts).strip()

@st.cache_data(ttl=86400, show_spinner=False)
def translate_botanical_text(value, target):
    if value is None:
        return value
    original = str(value).strip()
    if not original or original in ("—", "nan", "None") or target == "en":
        return original
    try:
        chunks = _split_text_chunks(original, max_len=1300)
        return "\n".join(_google_translate_chunk(c, target) for c in chunks)
    except Exception:
        return original

@st.cache_resource(show_spinner=False)
def load_assets():
    base_dir = Path(__file__).resolve().parent
    with open(base_dir / "class_names_1600.json", "r", encoding="utf-8") as f:
        all_species = json.load(f)

    model_mob = tf.keras.models.load_model(str(base_dir / "best_species_1600_inference.keras"), compile=False)

    eff_path = base_dir / "best_effnetv2_1600_inference.keras"
    model_eff = tf.keras.models.load_model(str(eff_path), compile=False) if eff_path.exists() else None

    species_df = pd.read_csv(base_dir / "species_info.csv")
    species_lookup = species_df.drop_duplicates(subset=["clean_scientific_name"]).set_index("clean_scientific_name").to_dict(orient="index")

    ref_urls = {}
    ref_file = base_dir / "reference_urls_1600.json"
    if ref_file.exists():
        with open(ref_file, "r", encoding="utf-8") as f:
            ref_urls = json.load(f)

    grad_model = None
    try:
        backbone = None
        for layer in model_mob.layers:
            if isinstance(layer, tf.keras.Model) and "mobilenet" in layer.name.lower():
                backbone = layer
                break
        if backbone is not None:
            last_conv = None
            for l in reversed(backbone.layers):
                if isinstance(l, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
                    last_conv = l.name
                    break
            if last_conv:
                grad_model = tf.keras.Model(
                    inputs=backbone.inputs,
                    outputs=[backbone.get_layer(last_conv).output, backbone.output]
                )
    except Exception:
        grad_model = None

    return model_mob, model_eff, all_species, species_lookup, ref_urls, grad_model

model_1600, model_effnet, all_species, species_lookup, ref_urls_map, gradcam_backbone = load_assets()

def generate_gradcam_overlay(img: Image.Image, top_class_idx: int):
    try:
        orig_rgb = img.convert("RGB")
        resized = orig_rgb.resize((192, 192))
        inp = preprocess_input(np.expand_dims(np.array(resized, dtype=np.float32), axis=0))

        if gradcam_backbone is not None:
            inp_tensor = tf.convert_to_tensor(inp)
            with tf.GradientTape() as tape:
                conv_outputs, features = gradcam_backbone(inp_tensor, training=False)
                x = features
                for layer in model_1600.layers:
                    if isinstance(layer, tf.keras.Model) or isinstance(layer, tf.keras.layers.InputLayer):
                        continue
                    x = layer(x, training=False)
                class_score = x[:, top_class_idx]
            grads = tape.gradient(class_score, conv_outputs)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            conv_map = conv_outputs[0]
            heatmap = tf.reduce_sum(conv_map * pooled_grads, axis=-1).numpy()
            heatmap = np.maximum(heatmap, 0)
            if np.max(heatmap) > 0:
                heatmap /= np.max(heatmap)
        else:
            arr = np.array(resized, dtype=np.float32) / 255.0
            green_focus = np.clip(arr[:, :, 1] - 0.5 * (arr[:, :, 0] + arr[:, :, 2]), 0, 1)
            heatmap = green_focus / (np.max(green_focus) + 1e-8)

        heat_img = Image.fromarray(np.uint8(heatmap * 255)).resize(orig_rgb.size, resample=Image.Resampling.BILINEAR)
        heat_arr = np.array(heat_img, dtype=np.float32) / 255.0
        colored = (cm.jet(heat_arr)[:, :, :3] * 255).astype(np.uint8)
        overlay = Image.blend(orig_rgb, Image.fromarray(colored), alpha=0.42)

        buf_orig = _io.BytesIO()
        orig_rgb.save(buf_orig, format="JPEG", quality=88)
        buf_heat = _io.BytesIO()
        overlay.save(buf_heat, format="JPEG", quality=88)
        return (
            base64.b64encode(buf_orig.getvalue()).decode("utf-8"),
            base64.b64encode(buf_heat.getvalue()).decode("utf-8"),
        )
    except Exception:
        return None, None

@st.cache_data(ttl=86400, show_spinner=False)
def get_reference_photo_url(scientific_name: str):
    if scientific_name in ref_urls_map and ref_urls_map[scientific_name]:
        return ref_urls_map[scientific_name]
    try:
        q = urllib.parse.quote(scientific_name)
        url = f"https://api.inaturalist.org/v1/taxa?q={q}&per_page=1"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        results = data.get("results", [])
        if results and results[0].get("default_photo"):
            return results[0]["default_photo"].get("medium_url", "")
    except Exception:
        pass
    return ""

def predict_top5_ensemble(img: Image.Image):
    rgb = img.convert("RGB")

    # 1. MobileNetV2 (192x192 + TTA)
    img_192 = rgb.resize((192, 192))
    flip_192 = ImageOps.mirror(img_192)
    batch_192 = np.stack([
        preprocess_input(np.array(img_192, dtype=np.float32)),
        preprocess_input(np.array(flip_192, dtype=np.float32)),
    ], axis=0)
    probs_mob = np.mean(model_1600.predict(batch_192, verbose=0), axis=0)

    # 2. EfficientNetV2-B0 (224x224 + TTA)
    if model_effnet is not None:
        img_224 = rgb.resize((224, 224))
        flip_224 = ImageOps.mirror(img_224)
        batch_224 = np.stack([
            np.array(img_224, dtype=np.float32),
            np.array(flip_224, dtype=np.float32),
        ], axis=0)
        probs_eff = np.mean(model_effnet.predict(batch_224, verbose=0), axis=0)
        # التجميع الموزون المثالي (0.70 MobileNetV2 + 0.30 EfficientNetV2-B0)
        probs = 0.70 * probs_mob + 0.30 * probs_eff
    else:
        probs = probs_mob

    top5_idx = np.argsort(probs)[::-1][:5]
    top5 = [(all_species[i], float(probs[i])) for i in top5_idx]

    eps = 1e-12
    entropy = -float(np.sum(probs * np.log(probs + eps)))
    norm_entropy = entropy / float(np.log(len(probs)))
    max_prob = float(probs[top5_idx[0]])
    is_ood = (max_prob < 0.14) or (norm_entropy > 0.79 and max_prob < 0.24)

    return top5, int(top5_idx[0]), is_ood, max_prob, norm_entropy

@st.cache_data(ttl=86400, show_spinner=False)
def get_chemical_compounds(scientific_name: str):
    if not scientific_name:
        return []
    try:
        encoded_name = urllib.parse.quote(scientific_name)
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded_name}/JSON?MaxRecords=5"
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(request, timeout=8) as response:
            data = json.loads(response.read().decode("utf-8"))
        compounds = data.get("PC_Compounds", [])
        names = []
        for compound in compounds[:5]:
            for prop in compound.get("props", []):
                urn = prop.get("urn", {})
                if urn.get("label") == "IUPAC Name":
                    value = prop.get("value", {}).get("sval", "")
                    if value:
                        names.append(value)
                    break
        return names[:5]
    except Exception:
        return []

def clean_val(val, default="—"):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    s = str(val).strip()
    return s if s and s.lower() not in ("nan", "none", "null") else default

def save_correction_to_github(wrong: str, correct: str, img_bytes: bytes):
    token = st.secrets.get("GITHUB_TOKEN", "")
    repo = st.secrets.get("GITHUB_REPO", "")
    if not token or not repo:
        return
    img = Image.open(_io.BytesIO(img_bytes)).convert("RGB").resize((224, 224))
    buf = _io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    fname = f"training_corrections/{int(time.time())}.jpg"
    req = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/contents/{fname}",
        data=json.dumps({
            "message": f"correction (1600 Ensemble): {wrong} -> {correct}",
            "content": b64
        }).encode(),
        method="PUT",
        headers={
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json"
        }
    )
    try:
        with urllib.request.urlopen(req) as r:
            img_url = json.loads(r.read())["content"]["html_url"]
    except Exception:
        img_url = None

    body = (
        f"**Ensemble Prediction:** {wrong}\n\n"
        f"**User Correct Label:** {correct}\n\n"
        + (f"[📷 Plant Image]({img_url})\n\n" if img_url else "")
    )
    req2 = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/issues",
        data=json.dumps({
            "title": f"1600 Ensemble Correction: {wrong} -> {correct}",
            "body": body,
            "labels": ["correction"]
        }).encode(),
        method="POST",
        headers={
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json"
        }
    )
    with urllib.request.urlopen(req2) as r:
        return json.loads(r.read())["html_url"]

for k in ("top5", "feedback_sent", "img_bytes", "chemicals", "ref_photos", "is_ood", "ood_override", "gradcam_pair"):
    if k not in st.session_state:
        st.session_state[k] = None

page_dir = "rtl" if LANG == "ar" else "ltr"

st.markdown(f"""
<div class="hero-box" dir="{page_dir}">
  <h1 style="margin:0 0 10px 0;font-size:2.2rem;font-weight:800;color:white;text-shadow:0 2px 10px rgba(0,0,0,0.25);">🌿 {tr('hero')}</h1>
  <p style="margin:0 0 12px 0;font-size:1.05rem;color:#ecfdf5;opacity:.95;">{tr('subtitle')}</p>
  <span class="hero-badge">{tr('badge_info')}</span>
</div>
""", unsafe_allow_html=True)

col_results, col_input = st.columns([2.25, 1], gap="large")

with col_input:
    st.markdown(f'<div dir="{page_dir}"><h4>{tr("input")}</h4></div>', unsafe_allow_html=True)
    tab_up, tab_cam = st.tabs([tr("upload_tab"), tr("camera_tab")])
    with tab_up:
        uploaded = st.file_uploader(
            tr("upload"),
            type=["jpg", "jpeg", "png", "webp", "jfif"],
            label_visibility="collapsed"
        )
    with tab_cam:
        camera = st.camera_input(tr("capture"), label_visibility="collapsed")
    img_source = uploaded or camera

    if img_source:
        image = Image.open(img_source)
        image = ImageOps.exif_transpose(image)
        prev_buf = _io.BytesIO()
        image.convert("RGB").save(prev_buf, format="JPEG", quality=90)
        prev_b64 = base64.b64encode(prev_buf.getvalue()).decode("utf-8")
        st.markdown(f"""
        <div class="preview-wrapper">
          <div class="ready-pill">{tr("ready_img")}</div>
          <img src="data:image/jpeg;base64,{prev_b64}" alt="Plant Preview" />
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:4px;'></div>", unsafe_allow_html=True)
    if st.button(tr("analyse"), key="analyse_btn", width="stretch"):
        if img_source is None:
            st.warning(tr("upload_first"))
        else:
            image = Image.open(img_source)
            image = ImageOps.exif_transpose(image)
            buf = _io.BytesIO()
            image.convert("RGB").save(buf, format="JPEG", quality=90)
            st.session_state.img_bytes = buf.getvalue()

            with st.spinner(tr("running")):
                top5, top1_idx, is_ood, max_prob, norm_entropy = predict_top5_ensemble(image)
                st.session_state.top5 = top5
                st.session_state.is_ood = is_ood
                st.session_state.ood_override = False
                st.session_state.feedback_sent = False
                st.session_state.gradcam_pair = generate_gradcam_overlay(image, top1_idx)

            with st.spinner(tr("fetching")):
                st.session_state.chemicals = get_chemical_compounds(top5[0][0])
                st.session_state.ref_photos = [
                    (sp_name, prob, get_reference_photo_url(sp_name))
                    for sp_name, prob in top5[:3]
                ]

with col_results:
    if st.session_state.top5 is not None:
        top5 = st.session_state.top5
        best_sp, best_prob = top5[0]
        info = species_lookup.get(best_sp, {})

        if st.session_state.is_ood and not st.session_state.ood_override:
            st.markdown(
                f'<div class="ood-alert-box" dir="{page_dir}">'
                f'<div style="font-size:1.12rem;font-weight:800;margin-bottom:8px;">{tr("ood_title")}</div>'
                f'<div style="font-size:0.95rem;line-height:1.65;">{tr("ood_body")}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if st.button(tr("ood_override"), key="ood_override_btn", width="stretch"):
                st.session_state.ood_override = True
                st.rerun()
        else:
            bars_parts = []
            for idx, (sp_name, prob) in enumerate(top5):
                fill_cls = "bar-fill-top" if idx == 0 else "bar-fill-sub"
                width_pct = max(prob * 100.0, 2.5)
                bars_parts.append(
                    f'<div class="bar-wrap">'
                    f'<div class="bar-header"><span dir="ltr"><i>{html.escape(sp_name)}</i></span><span dir="ltr" style="color:#4ade80;">{prob*100:.1f}%</span></div>'
                    f'<div class="bar-track"><div class="{fill_cls}" style="width:{width_pct:.1f}%"></div></div>'
                    f'</div>'
                )

            st.markdown(
                f'<div class="model-card-primary" dir="{page_dir}">'
                f'<div style="font-size:1.15rem;font-weight:800;color:#f8fafc;margin-bottom:12px;">{tr("top5_title")}</div>'
                f'{"".join(bars_parts)}'
                f'</div>',
                unsafe_allow_html=True,
            )

            orig_b64, heat_b64 = st.session_state.gradcam_pair or (None, None)
            if orig_b64 and heat_b64:
                st.markdown(
                    f'<div dir="{page_dir}" class="plant-section-heading">'
                    f'<span class="section-mark">🔍</span>{html.escape(tr("gradcam_title"))}</div>',
                    unsafe_allow_html=True,
                )
                gc1, gc2 = st.columns(2, gap="medium")
                with gc1:
                    st.markdown(
                        f'<div class="ref-card">'
                        f'<img src="data:image/jpeg;base64,{orig_b64}" alt="Original"/>'
                        f'<div class="ref-caption">{html.escape(tr("gradcam_orig"))}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                with gc2:
                    st.markdown(
                        f'<div class="ref-card">'
                        f'<img src="data:image/jpeg;base64,{heat_b64}" alt="Grad-CAM"/>'
                        f'<div class="ref-caption">{html.escape(tr("gradcam_heat"))}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                st.markdown(
                    f'<div dir="{page_dir}" style="font-size:0.86rem;color:#94a3b8;margin-bottom:12px;">'
                    f'{html.escape(tr("gradcam_desc"))}</div>',
                    unsafe_allow_html=True,
                )

            ref_photos = st.session_state.ref_photos or []
            valid_refs = [r for r in ref_photos if r[2]]
            if valid_refs:
                st.markdown(
                    f'<div dir="{page_dir}" class="plant-section-heading">'
                    f'<span class="section-mark">🖼️</span>{html.escape(tr("ref_gallery"))}</div>',
                    unsafe_allow_html=True,
                )
                ref_cols = st.columns(len(valid_refs), gap="medium")
                for idx, (sp_name, prob, ref_url) in enumerate(valid_refs):
                    with ref_cols[idx]:
                        st.markdown(
                            f'<div class="ref-card">'
                            f'<img src="{html.escape(ref_url)}" alt="{html.escape(sp_name)}"/>'
                            f'<div class="ref-caption" dir="ltr"><i>{html.escape(sp_name)}</i> ({prob*100:.1f}%)</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

            sahara_doc = bool(info.get("documented_in_morocco_sahara", False))
            sahara_toxic = bool(info.get("is_toxic_morocco_study", False))
            covid_doc = bool(info.get("documented_in_morocco_national_covid", False))
            covid_toxic = bool(info.get("covid_toxic", False))
            hazards_raw = clean_val(info.get("Known Hazards"), "")

            toxic_msgs = []
            if sahara_toxic:
                toxic_msgs.append("• دراسة الصحراء المغربية: مصنفة كنبتة سامة في الاستعمال التقليدي المحلي.")
            if covid_toxic:
                toxic_msgs.append("• الدراسة الوطنية المغربية: ورد تحذير من سميتها.")
            if hazards_raw:
                hazards_tr = translate_botanical_text(hazards_raw, LANG)
                toxic_msgs.append(f"• PFAF: {hazards_tr}")

            if toxic_msgs:
                joined_tox = "<br>".join(html.escape(m) for m in toxic_msgs)
                st.markdown(
                    f'<div class="toxic-alert-box" dir="{page_dir}">'
                    f'<div style="font-weight:800;font-size:1.02rem;margin-bottom:6px;">{tr("toxic_banner")}</div>'
                    f'<div style="font-size:0.93rem;line-height:1.65;">{joined_tox}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(f'<div class="safe-alert-box" dir="{page_dir}">{tr("safe_banner")}</div>', unsafe_allow_html=True)

            st.markdown(
                f'<div dir="{page_dir}" class="plant-section-heading">'
                f'<span class="section-mark">🌿</span>{html.escape(tr("plant_id"))}</div>',
                unsafe_allow_html=True,
            )

            vernacular_parts = []
            if sahara_doc:
                v_s = clean_val(info.get("moroccan_vernacular_name"), "")
                if v_s:
                    vernacular_parts.append(f"{v_s} (الصحراء)")
            if covid_doc:
                v_c = clean_val(info.get("covid_vernacular"), "")
                if v_c:
                    vernacular_parts.append(f"{v_c} (وطني)")
            vernacular_str = " | ".join(vernacular_parts) if vernacular_parts else "—"

            common_raw = clean_val(info.get("Common Name") or info.get("common_name"))
            common_tr = translate_botanical_text(common_raw, LANG) if common_raw != "—" else "—"

            id_c1, id_c2 = st.columns(2, gap="medium")
            with id_c1:
                st.markdown(
                    f'<div dir="{page_dir}" class="plant-info-card">'
                    f'<div class="plant-info-title">{tr("scientific")}</div>'
                    f'<div class="plant-info-value"><i dir="ltr">{html.escape(best_sp)}</i></div></div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div dir="{page_dir}" class="plant-info-card">'
                    f'<div class="plant-info-title">{tr("vernacular")}</div>'
                    f'<div class="plant-info-value">{html.escape(vernacular_str)}</div></div>',
                    unsafe_allow_html=True,
                )
            with id_c2:
                st.markdown(
                    f'<div dir="{page_dir}" class="plant-info-card">'
                    f'<div class="plant-info-title">{tr("common")}</div>'
                    f'<div class="plant-info-value">{html.escape(common_tr)}</div></div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div dir="{page_dir}" class="plant-info-card">'
                    f'<div class="plant-info-title">{tr("confidence")}</div>'
                    f'<div class="plant-info-value"><span dir="ltr" style="color:#4ade80;font-weight:800;">{best_prob*100:.1f}%</span></div></div>',
                    unsafe_allow_html=True,
                )

            st.markdown(
                f'<div dir="{page_dir}" class="plant-section-heading">'
                f'<span class="section-mark">🇲🇦</span>{html.escape(tr("morocco_sec"))}</div>',
                unsafe_allow_html=True,
            )
            if sahara_doc or covid_doc:
                m_cols = st.columns(2 if (sahara_doc and covid_doc) else 1, gap="medium")
                col_i = 0
                if sahara_doc:
                    s_txt = (
                        f"الجزء المستعمل: {clean_val(info.get('part_used_morocco'))}\n"
                        f"التحضير: {clean_val(info.get('preparation_morocco'))}\n"
                        f"الاستعمال العلاجي: {clean_val(info.get('therapeutic_use_morocco'))}"
                    )
                    with m_cols[col_i]:
                        st.markdown(
                            f'<div dir="{page_dir}" class="plant-info-card">'
                            f'<div class="plant-info-title">{tr("sahara_lbl")}</div>'
                            f'<div class="plant-info-value">{html.escape(s_txt)}</div></div>',
                            unsafe_allow_html=True,
                        )
                    col_i += 1
                if covid_doc:
                    c_txt = (
                        f"الجزء المستعمل: {clean_val(info.get('covid_part'))}\n"
                        f"طريقة الاستعمال: {clean_val(info.get('covid_use_mode'))}\n"
                        f"عدد الاستشهادات: {clean_val(info.get('covid_ntc'))}"
                    )
                    with m_cols[col_i]:
                        st.markdown(
                            f'<div dir="{page_dir}" class="plant-info-card">'
                            f'<div class="plant-info-title">{tr("covid_lbl")}</div>'
                            f'<div class="plant-info-value">{html.escape(c_txt)}</div></div>',
                            unsafe_allow_html=True,
                        )
            else:
                st.markdown(
                    f'<div dir="{page_dir}" class="plant-info-card">'
                    f'<div class="plant-info-value">{html.escape(tr("not_morocco"))}</div></div>',
                    unsafe_allow_html=True,
                )

            st.markdown(
                f'<div dir="{page_dir}" class="plant-section-heading">'
                f'<span class="section-mark">📋</span>{html.escape(tr("plant_info"))}</div>',
                unsafe_allow_html=True,
            )
            fam_val = clean_val(info.get("Family"))
            med_r = clean_val(info.get("Medicinal Rating"))
            edib_r = clean_val(info.get("Edibility Rating"))
            range_val = translate_botanical_text(clean_val(info.get("Native Range")), LANG)
            props_val = translate_botanical_text(clean_val(info.get("Medicinal Properties")), LANG)

            p_c1, p_c2 = st.columns(2, gap="medium")
            with p_c1:
                st.markdown(
                    f'<div dir="{page_dir}" class="plant-info-card">'
                    f'<div class="plant-info-title">{tr("family")}</div>'
                    f'<div class="plant-info-value" dir="ltr">{html.escape(fam_val)}</div></div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div dir="{page_dir}" class="plant-info-card">'
                    f'<div class="plant-info-title">{tr("range")}</div>'
                    f'<div class="plant-info-value">{html.escape(range_val)}</div></div>',
                    unsafe_allow_html=True,
                )
            with p_c2:
                st.markdown(
                    f'<div dir="{page_dir}" class="plant-info-card">'
                    f'<div class="plant-info-title">{tr("med_rating")} &bull; {tr("edib_rating")}</div>'
                    f'<div class="plant-info-value" dir="ltr">Medicinal: {html.escape(med_r)} / 5  |  Edibility: {html.escape(edib_r)} / 5</div></div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div dir="{page_dir}" class="plant-info-card">'
                    f'<div class="plant-info-title">{tr("properties")}</div>'
                    f'<div class="plant-info-value">{html.escape(props_val[:1800])}</div></div>',
                    unsafe_allow_html=True,
                )

            chemicals = st.session_state.chemicals or []
            if chemicals:
                st.markdown(
                    f'<div dir="{page_dir}" class="plant-section-heading">'
                    f'<span class="section-mark">🧪</span>{html.escape(tr("chemicals"))}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<div dir="{page_dir}" class="chemical-panel">'
                    f'<div style="color:#94a3b8;font-size:.86rem;margin-bottom:8px;">{html.escape(tr("chem_note"))}</div>'
                    + " ".join(f'<span dir="ltr" class="chemical-pill">{html.escape(str(n))}</span>' for n in chemicals)
                    + '</div>',
                    unsafe_allow_html=True,
                )

            st.markdown(
                f'<div class="feedback-box" dir="{page_dir}">'
                f'<div style="font-size:1.05rem;font-weight:700;color:#4ade80;margin-bottom:4px;">{tr("evaluate")}</div>'
                f'<div style="font-size:0.92rem;color:#cbd5e1;">{tr("evaluate_help")}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

            if not st.session_state.feedback_sent:
                fc1, fc2 = st.columns(2, gap="medium")
                with fc1:
                    if st.button(tr("yes"), width="stretch"):
                        st.session_state.feedback_sent = "correct"
                        st.rerun()
                with fc2:
                    if st.button(tr("no"), width="stretch", key="wrong_btn_trigger"):
                        st.session_state.feedback_sent = "wrong"
                        st.rerun()
            elif st.session_state.feedback_sent == "correct":
                st.success(tr("thanks"))
            elif st.session_state.feedback_sent == "wrong":
                st.markdown(f'<div dir="{page_dir}"><b>{tr("select_true")}</b></div>', unsafe_allow_html=True)
                other_candidates = [""] + [sp for sp, _ in top5[1:]]
                chosen_cand = st.selectbox("Candidates", other_candidates, label_visibility="collapsed")
                typed_sp = st.text_input(tr("custom_sp"))
                final_true = typed_sp.strip() if typed_sp.strip() else chosen_cand
                if final_true and st.button(tr("submit"), width="stretch"):
                    with st.spinner(tr("saving")):
                        try:
                            save_correction_to_github(best_sp, final_true, st.session_state.img_bytes)
                            st.session_state.feedback_sent = "done"
                            st.rerun()
                        except Exception as e:
                            st.error(tr("save_error") + str(e))
            elif st.session_state.feedback_sent == "done":
                st.success(tr("saved"))
    else:
        st.markdown(
            f'<div dir="{page_dir}" style="display:flex;flex-direction:column;align-items:center;justify-content:center;'
            f'height:380px;background:rgba(15,23,42,0.55);border:1px dashed rgba(148,163,184,0.25);'
            f'border-radius:22px;color:#94a3b8;text-align:center;padding:20px;">'
            f'<div style="font-size:2.8rem;margin-bottom:12px;">🌱</div>'
            f'<div style="font-size:1.15rem;font-weight:700;color:#e2e8f0;">{tr("ready")}</div>'
            f'<div style="font-size:0.95rem;margin-top:6px;">{tr("ready_help")}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.markdown(
    f'<div dir="{page_dir}" style="text-align:center;font-size:.85rem;color:#94a3b8;margin-top:40px;'
    f'border-top:1px solid rgba(255,255,255,0.08);padding-top:18px;">'
    f'{tr("academic")}<br>'
    f'{tr("dataset")} • <b>Version 3.0 (Ensemble AI)</b>'
    f'</div>',
    unsafe_allow_html=True,
)
