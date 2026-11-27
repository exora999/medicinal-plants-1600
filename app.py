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
        "badge_info": "🔬 1,600 Species &bull; Ensemble AI &bull; Smart Search &bull; Morocco Map &bull; PFAF &bull; PubChem",
        "mode_photo": "📸 Identify by Photo (AI)",
        "mode_search": "🔎 Search by Name or Use",
        "search_header": "🔎 Smart Botanical & Moroccan Vernacular Search",
        "search_placeholder": "Type Moroccan Darija name (e.g., أزير, فليو, الشيبة), scientific name, or medicinal use (e.g., cough, antiseptic)...",
        "search_filter": "Filter Scope:",
        "filter_all": "🌍 All 1,600 Species",
        "filter_morocco": "🇲🇦 Documented in Morocco Only",
        "search_matches": "Matching Species Found:",
        "search_select": "Select a species to view its full botanical card, reference photos & Morocco map:",
        "search_empty": "No matching species found. Try another keyword (e.g., Thymus, Mentha, أزير, الزعتر, fever).",
        "map_title": "🗺️ Geographical Distribution in Morocco",
        "map_regions_lbl": "Detected Ecological Regions:",
        "map_general_med": "Mediterranean / North African Flora (Widespread across Moroccan plains & foothills)",
        "map_not_native": "Not native to Morocco (Cultivated or exotic species outside Moroccan flora)",
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
        "badge_info": "🔬 1 600 Espèces &bull; IA Ensemble &bull; Recherche Inverse &bull; Carte Maroc &bull; PFAF &bull; PubChem",
        "mode_photo": "📸 Identifier par Photo (IA)",
        "mode_search": "🔎 Rechercher par Nom ou Usage",
        "search_header": "🔎 Recherche Botanique & Vernaculaire Marocaine",
        "search_placeholder": "Tapez un nom marocain (ex. أزير, فليو, الشيبة), nom scientifique, ou usage médical (ex. cough, antiseptic)...",
        "search_filter": "Portée du filtre :",
        "filter_all": "🌍 Toutes les 1 600 Espèces",
        "filter_morocco": "🇲🇦 Documentées au Maroc uniquement",
        "search_matches": "Espèces correspondantes :",
        "search_select": "Sélectionnez une espèce pour afficher sa fiche complète, ses photos et sa carte au Maroc :",
        "search_empty": "Aucune espèce trouvée. Essayez un autre mot-clé (ex. Thymus, Mentha, أزير, الزعتر).",
        "map_title": "🗺️ Répartition Géographique au Maroc",
        "map_regions_lbl": "Régions écologiques détectées :",
        "map_general_med": "Flore méditerranéenne / nord-africaine (Répandue dans les plaines et piémonts du Maroc)",
        "map_not_native": "Non indigène au Maroc (Espèce cultivée ou exotique hors flore marocaine)",
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
        "ood_body": "Cette image ne correspond clairement à aucune des 1 600 espèces entraînées. Essayez une photo nette des feuilles ou fleurs.",
        "ood_override": "👁️ Afficher quand même l’espèce la plus proche",
        "top5_title": "🎯 Top 5 des espèces prédites (Ensemble Double Modèle)",
        "gradcam_title": "🔍 Attention visuelle de l’IA (Grad-CAM)",
        "gradcam_orig": "Image d’origine",
        "gradcam_heat": "Zones de décision (Feuilles / Fleurs)",
        "gradcam_desc": "Les zones chaudes (rouge/jaune) montrent les structures botaniques utilisées par le réseau de neurones.",
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
        "academic": "⚠️ <b>Projet académique et éducatif</b> — N’utilisez PAS cet outil pour un diagnostic médical.",
        "dataset": "Base de données : 1 600 Espèces • 73 948 Images (iNaturalist + PFAF + Études Marocaines)"
    },
    "ar": {
        "hero": "نظام التعرّف على النباتات الطبية",
        "subtitle": "الذكاء الاصطناعي التجميعي المزدوج (MobileNetV2 + EfficientNetV2) والسياق المغربي — <b>1,600 نوع نباتي</b>",
        "badge_info": "🔬 1,600 نوع &bull; دمج نموذجين &bull; بحث بالاسم/المرض &bull; خريطة المغرب &bull; PFAF &bull; PubChem",
        "mode_photo": "📸 التعرّف بالصورة (ذكاء اصطناعي)",
        "mode_search": "🔎 البحث بالاسم الدارج أو العالمي",
        "search_header": "🔎 البحث الذكي بالاسم المغربي أو الاستعمال العلاجي",
        "search_placeholder": "اكتب الاسم بالدارجة (مثل: أزير، فليو، الشيبة، الزعتر، الحرمل) أو الاسم العلمي أو المرض (مثل: cough, stomach)...",
        "search_filter": "نطاق البحث:",
        "filter_all": "🌍 جميع الأنواع (1,600 نوع)",
        "filter_morocco": "🇲🇦 النباتات الموثقة في المغرب فقط",
        "search_matches": "الأنواع المطابقة:",
        "search_select": "اختر النبتة لعرض بطاقتها العلمية وصورها وخريطة انتشارها في المغرب:",
        "search_empty": "لم يتم العثور على نتائج مطابقة. جرّب كلمة أخرى (مثل: أزير، الزعتر، Thymus، Mentha، antiseptic).",
        "map_title": "🗺️ خريطة الانتشار الجغرافي للنبتة في المغرب",
        "map_regions_lbl": "المناطق البيئية المرصودة:",
        "map_general_med": "نباتات حوض المتوسط وشمال إفريقيا (منتشرة في السهول والهضاب المغربية)",
        "map_not_native": "غير مستوطنة طبيعياً في المغرب (نبتة مزروعة أو خارج الغطاء النباتي المغربي)",
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
        "ood_body": "هذه الصورة لا تطابق بشكل موثوق أياً من الأنواع الـ 1,600 المدرَّبة. يُرجى رفع صورة أوضح للأوراق أو الأزهار.",
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
      padding: 26px 24px;
      text-align: center;
      color: white;
      margin-top: 8px;
      margin-bottom: 22px;
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
  }
  .model-card-primary {
      background: linear-gradient(180deg, rgba(15, 23, 42, 0.88) 0%, rgba(17, 24, 39, 0.95) 100%);
      backdrop-filter: blur(12px);
      border: 1.5px solid rgba(74, 222, 128, 0.45);
      border-radius: 22px;
      padding: 22px 22px;
      box-shadow: 0 12px 32px -10px rgba(0, 0, 0, 0.45);
      margin-bottom: 18px;
      transition: transform 0.25s ease, border-color 0.25s ease;
  }
  .model-card-primary:hover {
      transform: translateY(-3px);
      border-color: #4ade80;
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
      transition: transform 0.24s ease, border-color 0.24s ease, box-shadow 0.24s ease;
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
  }
  .safe-alert-box {
      background: rgba(15, 23, 42, 0.85);
      border: 1px solid rgba(52, 211, 153, 0.35);
      border-radius: 16px;
      padding: 13px 18px;
      margin: 14px 0;
      color: #a7f3d0;
      font-weight: 600;
  }
  .feedback-box {
      background: rgba(15, 23, 42, 0.78);
      border: 1px solid rgba(148, 163, 184, 0.22);
      border-radius: 18px;
      padding: 18px 22px;
      margin-top: 16px;
  }
  .morocco-map-card {
      background: linear-gradient(145deg, rgba(15, 23, 42, 0.94), rgba(6, 78, 59, 0.35));
      border: 1px solid rgba(52, 211, 153, 0.35);
      border-radius: 20px;
      padding: 18px;
      margin: 10px 0 18px 0;
      box-shadow: 0 12px 30px rgba(0, 0, 0, 0.35);
  }
  .region-pill-active {
      display: inline-block;
      background: rgba(16, 185, 129, 0.22);
      border: 1px solid #34d399;
      color: #a7f3d0;
      border-radius: 999px;
      padding: 5px 13px;
      margin: 4px 5px 4px 0;
      font-size: 0.82rem;
      font-weight: 700;
  }
  .region-pill-inactive {
      display: inline-block;
      background: rgba(30, 41, 59, 0.65);
      border: 1px solid rgba(148, 163, 184, 0.2);
      color: #94a3b8;
      border-radius: 999px;
      padding: 5px 13px;
      margin: 4px 5px 4px 0;
      font-size: 0.80rem;
  }

  .stButton>button {
      border-radius: 14px !important;
      font-weight: 700 !important;
      background: rgba(30, 41, 59, 0.88) !important;
      border: 1px solid rgba(148, 163, 184, 0.25) !important;
      color: #ffffff !important;
      padding: 11px 18px !important;
      transition: transform 0.18s ease, box-shadow 0.22s ease, background 0.22s ease, border-color 0.22s ease !important;
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
  }
  .st-key-wrong_btn_trigger button:hover {
      background: linear-gradient(135deg, #dc2626, #991b1b) !important;
      border-color: #f87171 !important;
      box-shadow: 0 12px 24px -4px rgba(220, 38, 38, 0.55) !important;
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
  }
  .st-key-analyse_btn button:hover {
      background: linear-gradient(90deg, #15803d 0%, #16a34a 50%, #059669 100%) !important;
      box-shadow: 0 14px 34px -2px rgba(74, 222, 128, 0.78) !important;
      transform: translateY(-3px) !important;
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
      padding: 9px 22px;
      color: #cbd5e1;
      font-weight: 700;
      transition: all 0.2s ease !important;
  }
  .stTabs [data-baseweb="tab"]:hover {
      border-color: rgba(74, 222, 128, 0.5) !important;
      color: #ffffff !important;
      transform: translateY(-2px);
  }
  .stTabs [aria-selected="true"] {
      background: linear-gradient(135deg, rgba(22, 163, 74, 0.42), rgba(15, 23, 42, 0.88)) !important;
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
      color: #f8fafc;
      font-size: 1.06rem;
      font-weight: 800;
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
      box-shadow: 0 10px 26px rgba(0, 0, 0, 0.28);
      transition: transform 0.24s ease, border-color 0.24s ease;
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
  }
  .plant-info-card:hover {
      transform: translateY(-4px);
      border-color: rgba(74, 222, 128, 0.65);
  }
  .plant-info-title {
      display: block;
      font-size: .79rem;
      color: #6ee7b7;
      font-weight: 800;
      margin-bottom: 7px;
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
  [dir="rtl"] .bar-wrap:hover {
      transform: translateX(-4px);
  }
</style>
""", unsafe_allow_html=True)

top_left, top_right = st.columns([3.1, 1.1])
with top_left:
    st.markdown(
        '<div class="top-lang-badge">🌿 Dual-Model Ensemble AI &bull; 1,600 Species &bull; Smart Search &bull; Morocco Map</div>',
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
    species_df = species_df.drop_duplicates(subset=["clean_scientific_name"]).copy()
    species_lookup = species_df.set_index("clean_scientific_name").to_dict(orient="index")

    # بناء فهرس بحث سريع يشمل الاسم العلمي والشائع والدارجة المغربية والاستعمالات الطبية
    search_cols = [
        "clean_scientific_name", "Common Name", "common_name", "Family",
        "moroccan_vernacular_name", "covid_vernacular",
        "therapeutic_use_morocco", "Medicinal Properties", "Native Range"
    ]
    existing_cols = [c for c in search_cols if c in species_df.columns]
    species_df["_search_blob"] = species_df[existing_cols].fillna("").astype(str).agg(" | ".join, axis=1).str.lower()

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

    return model_mob, model_eff, all_species, species_df, species_lookup, ref_urls, grad_model

model_1600, model_effnet, all_species, species_df, species_lookup, ref_urls_map, gradcam_backbone = load_assets()

def clean_val(val, default="—"):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    s = str(val).strip()
    return s if s and s.lower() not in ("nan", "none", "null") else default

def infer_morocco_regions(info: dict):
    """تحديد أقاليم الانتشار الجغرافي في المغرب انطلاقاً من الدراسات الميدانية والموطن الأصلي."""
    sahara_doc = bool(info.get("documented_in_morocco_sahara", False))
    covid_doc = bool(info.get("documented_in_morocco_national_covid", False))
    native_range = clean_val(info.get("Native Range"), "").lower()
    use_mor = clean_val(info.get("therapeutic_use_morocco"), "").lower()

    regions = set()
    if sahara_doc or any(k in native_range for k in ("sahara", "desert", "arid", "mauritania")):
        regions.add("sahara")
        regions.add("souss")
    if any(k in native_range for k in ("atlas", "mountain", "alpine", "montane")):
        regions.add("high_atlas")
        regions.add("middle_atlas")
    if any(k in native_range for k in ("mediterranean", "spain", "portugal", "iberia", "algeria", "tunisia", "north africa")):
        regions.add("rif")
        regions.add("atlantic_plains")
        regions.add("middle_atlas")
    if "morocco" in native_range or "maroc" in native_range:
        regions.update(["high_atlas", "souss", "rif", "atlantic_plains"])
    if covid_doc and not regions:
        regions.update(["rif", "atlantic_plains", "middle_atlas", "high_atlas", "souss"])
    if "souss" in use_mor or "anti-atlas" in native_range:
        regions.add("souss")

    is_present = len(regions) > 0
    return is_present, regions

def render_morocco_distribution_map(info: dict, lang: str, page_dir: str):
    is_present, active_regions = infer_morocco_regions(info)

    region_labels = {
        "rif": {"ar": "الريف والشمال", "fr": "Rif & Nord", "en": "Rif & North"},
        "atlantic_plains": {"ar": "السهول الأطلسية والهضاب", "fr": "Plaines Atlantiques", "en": "Atlantic Plains"},
        "middle_atlas": {"ar": "الأطلس المتوسط والشرق", "fr": "Moyen Atlas & Oriental", "en": "Middle Atlas & East"},
        "high_atlas": {"ar": "الأطلس الكبير", "fr": "Haut Atlas", "en": "High Atlas"},
        "souss": {"ar": "سوس ماسة والأطلس الصغير", "fr": "Souss-Massa & Anti-Atlas", "en": "Souss-Massa & Anti-Atlas"},
        "sahara": {"ar": "الأقاليم الجنوبية والصحراء", "fr": "Sahara Marocain", "en": "Moroccan Sahara"},
    }

    def r_style(code):
        if code in active_regions:
            return 'fill="#10b981" fill-opacity="0.72" stroke="#6ee7b7" stroke-width="2"'
        return 'fill="#1e293b" fill-opacity="0.75" stroke="#475569" stroke-width="1.2"'

    svg_map = f"""
    <svg viewBox="0 0 360 360" style="width:100%;max-width:310px;height:auto;display:block;margin:0 auto;filter:drop-shadow(0 8px 18px rgba(0,0,0,0.45));">
      <!-- البحر والمحيط -->
      <rect x="0" y="0" width="360" height="360" rx="18" fill="#061224" />
      <!-- الريف والشمال -->
      <path d="M 215,24 L 268,32 L 282,58 L 235,68 L 204,55 Z" {r_style("rif")} />
      <!-- السهول الأطلسية -->
      <path d="M 204,55 L 235,68 L 222,122 L 168,136 L 178,92 Z" {r_style("atlantic_plains")} />
      <!-- الأطلس المتوسط والشرق -->
      <path d="M 235,68 L 282,58 L 315,105 L 275,142 L 222,122 Z" {r_style("middle_atlas")} />
      <!-- الأطلس الكبير -->
      <path d="M 168,136 L 222,122 L 275,142 L 245,176 L 154,168 Z" {r_style("high_atlas")} />
      <!-- سوس ماسة والأطلس الصغير -->
      <path d="M 154,168 L 245,176 L 218,215 L 132,208 L 142,182 Z" {r_style("souss")} />
      <!-- الصحراء المغربية -->
      <path d="M 132,208 L 218,215 L 178,262 L 118,332 L 58,332 L 94,256 Z" {r_style("sahara")} />
      <!-- أسماء مختصرة على الخريطة -->
      <text x="242" y="49" fill="#f8fafc" font-size="9" font-weight="700" text-anchor="middle">Rif</text>
      <text x="272" y="104" fill="#f8fafc" font-size="9" font-weight="700" text-anchor="middle">Moyen Atlas</text>
      <text x="215" y="152" fill="#f8fafc" font-size="9.5" font-weight="800" text-anchor="middle">Haut Atlas</text>
      <text x="178" y="194" fill="#f8fafc" font-size="9" font-weight="700" text-anchor="middle">Souss-Massa</text>
      <text x="135" y="274" fill="#f8fafc" font-size="10" font-weight="800" text-anchor="middle">Sahara</text>
    </svg>
    """

    pills_html = []
    for code, names in region_labels.items():
        cls = "region-pill-active" if code in active_regions else "region-pill-inactive"
        prefix = "✅ " if code in active_regions else "• "
        pills_html.append(f'<span class="{cls}">{prefix}{html.escape(names[lang])}</span>')

    status_note = tr("map_regions_lbl") if is_present else tr("map_not_native")

    st.markdown(
        f'<div dir="{page_dir}" class="plant-section-heading">'
        f'<span class="section-mark">🗺️</span>{html.escape(tr("map_title"))}</div>',
        unsafe_allow_html=True,
    )
    mc1, mc2 = st.columns([1.05, 1.45], gap="medium")
    with mc1:
        st.markdown(f'<div class="morocco-map-card">{svg_map}</div>', unsafe_allow_html=True)
    with mc2:
        st.markdown(
            f'<div dir="{page_dir}" class="morocco-map-card" style="height:calc(100% - 28px);display:flex;flex-direction:column;justify-content:center;">'
            f'<div style="font-size:0.95rem;font-weight:800;color:#6ee7b7;margin-bottom:10px;">{html.escape(status_note)}</div>'
            f'<div>{"".join(pills_html)}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

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
    img_192 = rgb.resize((192, 192))
    flip_192 = ImageOps.mirror(img_192)
    batch_192 = np.stack([
        preprocess_input(np.array(img_192, dtype=np.float32)),
        preprocess_input(np.array(flip_192, dtype=np.float32)),
    ], axis=0)
    probs_mob = np.mean(model_1600.predict(batch_192, verbose=0), axis=0)

    if model_effnet is not None:
        img_224 = rgb.resize((224, 224))
        flip_224 = ImageOps.mirror(img_224)
        batch_224 = np.stack([
            np.array(img_224, dtype=np.float32),
            np.array(flip_224, dtype=np.float32),
        ], axis=0)
        probs_eff = np.mean(model_effnet.predict(batch_224, verbose=0), axis=0)
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

def render_full_botanical_profile(best_sp: str, best_prob: float, page_dir: str, chemicals: list):
    info = species_lookup.get(best_sp, {})

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
        if best_prob is not None:
            st.markdown(
                f'<div dir="{page_dir}" class="plant-info-card">'
                f'<div class="plant-info-title">{tr("confidence")}</div>'
                f'<div class="plant-info-value"><span dir="ltr" style="color:#4ade80;font-weight:800;">{best_prob*100:.1f}%</span></div></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div dir="{page_dir}" class="plant-info-card">'
                f'<div class="plant-info-title">{tr("family")}</div>'
                f'<div class="plant-info-value" dir="ltr">{html.escape(clean_val(info.get("Family")))}</div></div>',
                unsafe_allow_html=True,
            )

    # خريطة الانتشار الجغرافي في المغرب
    render_morocco_distribution_map(info, LANG, page_dir)

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

mode_photo_tab, mode_search_tab = st.tabs([tr("mode_photo"), tr("mode_search")])

with mode_photo_tab:
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

                render_full_botanical_profile(best_sp, best_prob, page_dir, st.session_state.chemicals or [])

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

with mode_search_tab:
    st.markdown(
        f'<div dir="{page_dir}" class="plant-section-heading">'
        f'<span class="section-mark">🔎</span>{html.escape(tr("search_header"))}</div>',
        unsafe_allow_html=True,
    )

    sc1, sc2 = st.columns([2.4, 1.0], gap="medium")
    with sc1:
        search_query = st.text_input(
            "Search",
            placeholder=tr("search_placeholder"),
            key="reverse_search_query",
            label_visibility="collapsed"
        )
    with sc2:
        scope_opt = st.selectbox(
            "Scope",
            [tr("filter_morocco"), tr("filter_all")],
            index=0,
            key="reverse_search_scope",
            label_visibility="collapsed"
        )

    filtered_df = species_df.copy()
    if scope_opt == tr("filter_morocco"):
        m_mask = pd.Series(False, index=filtered_df.index)
        if "documented_in_morocco_sahara" in filtered_df.columns:
            m_mask = m_mask | filtered_df["documented_in_morocco_sahara"].fillna(False).astype(bool)
        if "documented_in_morocco_national_covid" in filtered_df.columns:
            m_mask = m_mask | filtered_df["documented_in_morocco_national_covid"].fillna(False).astype(bool)
        filtered_df = filtered_df[m_mask]

    q_clean = search_query.strip().lower()
    if q_clean:
        filtered_df = filtered_df[filtered_df["_search_blob"].str.contains(q_clean, regex=False, na=False)]

    if len(filtered_df) == 0:
        st.info(tr("search_empty"))
    else:
        st.markdown(
            f'<div dir="{page_dir}" style="font-size:0.9rem;color:#6ee7b7;font-weight:700;margin:8px 0;">'
            f'{html.escape(tr("search_matches"))} <span dir="ltr">{len(filtered_df)}</span></div>',
            unsafe_allow_html=True,
        )

        # بناء قائمة الاختيار بصيغة واضحة تجمع الاسم العلمي والدارجة المغربية
        display_options = []
        label_to_sp = {}
        for _, row in filtered_df.head(120).iterrows():
            sp_name = str(row["clean_scientific_name"])
            v1 = clean_val(row.get("moroccan_vernacular_name"), "")
            v2 = clean_val(row.get("covid_vernacular"), "")
            vern = " / ".join(dict.fromkeys([v for v in (v1, v2) if v]))
            lbl = f"{sp_name}  —  🇲🇦 {vern}" if vern else sp_name
            display_options.append(lbl)
            label_to_sp[lbl] = sp_name

        st.markdown(f'<div dir="{page_dir}" style="font-size:0.92rem;color:#cbd5e1;margin-bottom:4px;">{html.escape(tr("search_select"))}</div>', unsafe_allow_html=True)
        chosen_label = st.selectbox(
            "Select Species",
            display_options,
            key="reverse_search_selected_sp",
            label_visibility="collapsed"
        )

        selected_sp = label_to_sp[chosen_label]
        ref_url_single = get_reference_photo_url(selected_sp)
        if ref_url_single:
            _, img_mid, _ = st.columns([1, 1.4, 1])
            with img_mid:
                st.markdown(
                    f'<div class="ref-card">'
                    f'<img src="{html.escape(ref_url_single)}" style="height:230px;" alt="{html.escape(selected_sp)}"/>'
                    f'<div class="ref-caption" dir="ltr"><i>{html.escape(selected_sp)}</i> (iNaturalist Reference)</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        chems_single = get_chemical_compounds(selected_sp)
        render_full_botanical_profile(selected_sp, None, page_dir, chems_single)

st.markdown(
    f'<div dir="{page_dir}" style="text-align:center;font-size:.85rem;color:#94a3b8;margin-top:40px;'
    f'border-top:1px solid rgba(255,255,255,0.08);padding-top:18px;">'
    f'{tr("academic")}<br>'
    f'{tr("dataset")} • <b>Version 4.0 (Ensemble AI + Smart Search + Morocco Map)</b>'
    f'</div>',
    unsafe_allow_html=True,
)
