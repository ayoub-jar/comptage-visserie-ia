#!/usr/bin/env python3
"""
Application Web de comptage de visserie / écrous
Optimisée pour Tablettes, Smartphones et Streamlit Cloud
Moteur : OpenCV Watershed
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
)
from reportlab.lib.enums import TA_CENTER
import datetime
import io
import os
import tempfile

# Configuration responsive de la page
st.set_page_config(
    page_title="Contrôle Qualité - Vision Industrielle",
    page_icon="🔩",
    layout="centered"
)

# ─────────────────────────────────────────────
#  ALGORITHME WATERSHED INDUSTRIEL ROBUSTE
# ─────────────────────────────────────────────

def detect_and_count(image_cv, sensitivity: str = "medium"):
    """
    Algorithme de segmentation Watershed pour séparer et compter 
    les pièces même si elles se touchent ou sont vues de profil.
    """
    sensitivity_params = {
        "low":    dict(dist_thresh=0.55, min_area=800,  blur_size=5),
        "medium": dict(dist_thresh=0.38, min_area=350,  blur_size=5),
        "high":   dict(dist_thresh=0.20, min_area=100,  blur_size=3),
    }
    p = sensitivity_params[sensitivity]

    # Conversion en nuances de gris
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)

    # Filtrage des reflets métalliques brillants (Fermeture morphologique)
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    gray_cleaned = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel_clean)

    # Seuillage adaptatif sur fond blanc
    blurred = cv2.medianBlur(gray_cleaned, p["blur_size"])
    _, thresh = cv2.threshold(blurred, 242, 255, cv2.THRESH_BINARY_INV)

    # Nettoyage du bruit de fond
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh_cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open, iterations=2)

    # Séparation des objets collés via transformée de distance
    sure_bg = cv2.dilate(thresh_cleaned, kernel_open, iterations=3)
    dist_transform = cv2.distanceTransform(thresh_cleaned, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, p["dist_thresh"] * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Zone de contact inconnue
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marquage des composants
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # Application du Watershed
    annotated = image_cv.copy()
    markers = cv2.watershed(annotated, markers)
    
    count = 0
    unique_markers = np.unique(markers)
    
    for marker in unique_markers:
        if marker <= 1:
            continue
            
        mask = np.uint8(markers == marker)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            c = contours[0]
            area = cv2.contourArea(c)
            if area > p["min_area"]:
                count += 1
                x, y, w, h = cv2.boundingRect(c)
                # Dessin du rectangle de détection (Vert flashy)
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 210, 50), 2)
                cv2.putText(annotated, f"#{count}", (x + 3, y + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 210, 50), 2)

    # En-tête visuel avec le score total
    cv2.rectangle(annotated, (5, 5), (220, 50), (0, 0, 0), -1)
    cv2.putText(annotated, f"Total: {count}", (12, 38),
                cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 255, 150), 2)

    return count, annotated


# ─────────────────────────────────────────────
#  GÉNÉRATEUR DE RAPPORT PDF EN MÉMOIRE
# ─────────────────────────────────────────────

def make_pdf_bytes(result: dict):
    pdf_buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        pdf_buffer, pagesize=A4,
        rightMargin=2*cm, leftMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm
    )
    styles = getSampleStyleSheet()
    story = []

    header_style = ParagraphStyle("h", parent=styles["Title"], fontSize=18, textColor=colors.HexColor("#1a2e5a"))
    sub_style = ParagraphStyle("s", parent=styles["Normal"], fontSize=10, textColor=colors.grey, alignment=TA_CENTER)

    story.append(Paragraph("🔩 RAPPORT DE CONTRÔLE QUALITÉ MOBILE", header_style))
    story.append(Paragraph("Généré automatiquement par Vision Industrielle Cloud", sub_style))
    story.append(Spacer(1, 0.4*cm))
    story.append(Table([[""]], colWidths=[17*cm], style=TableStyle([("LINEBELOW", (0,0), (-1,-1), 2, colors.HexColor("#1a2e5a"))])))
    story.append(Spacer(1, 0.5*cm))

    now = datetime.datetime.now()
    info_data = [
        ["Date / Heure", now.strftime("%d/%m/%Y à %H:%M:%S")],
        ["Opérateur", result.get("operator", "—")],
        ["Référence lot", result.get("lot_ref", "—")],
        ["Type de pièce", result.get("piece_type", "—")],
        ["Quantité attendue", str(result.get("expected", "—"))],
    ]
    info_table = Table(info_data, colWidths=[6*cm, 11*cm])
    info_table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#eef2fa")),
        ("FONTNAME", (0,0), (0,-1), "Helvetica-Bold"),
        ("GRID", (0,0), (-1,-1), 0.5, colors.HexColor("#c5cfe0")),
        ("ROWBACKGROUNDS", (1,0), (1,-1), [colors.white, colors.HexColor("#f7f9fd")]),
        ("PADDING", (0,0), (-1,-1), 6),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 0.6*cm))

    count = result["count"]
    expected = result.get("expected")
    if expected and expected.isdigit():
        exp = int(expected)
        ok = count == exp
        verdict = "✅  CONFORME" if ok else "❌  NON CONFORME"
        v_color = colors.HexColor("#1e7e34") if ok else colors.HexColor("#c0392b")
        diff = count - exp
        ecart = f"Écart : {f'+{diff}' if diff > 0 else diff} pièce(s)" if not ok else "Aucun écart"
    else:
        verdict = f"✔  {count} élément(s) comptés"
        v_color = colors.HexColor("#1a2e5a")
        ecart = ""

    v_data = [[f"Éléments détectés : {count}   |   {verdict}"]]
    if ecart: v_data.append([ecart])
    vt = Table(v_data, colWidths=[17*cm])
    vt_styles = [("BACKGROUND", (0,0), (-1,0), v_color), ("TEXTCOLOR", (0,0), (-1,0), colors.white), ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"), ("ALIGN", (0,0), (-1,-1), "CENTER"), ("PADDING", (0,0), (-1,-1), 8)]
    if ecart: vt_styles += [("BACKGROUND", (0,1), (-1,1), colors.HexColor("#fdecea")), ("TEXTCOLOR", (0,1), (-1,1), colors.HexColor("#c0392b")), ("FONTNAME", (0,1), (-1,1), "Helvetica-Bold")]
    vt.setStyle(TableStyle(vt_styles))
    story.append(vt)
    story.append(Spacer(1, 0.6*cm))

    if "ann_img" in result:
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        result["ann_img"].save(tmp.name, "JPEG", quality=85)
        story.append(Paragraph("Image de contrôle (Vision Numérique)", styles["Heading2"]))
        story.append(Spacer(1, 0.2*cm))
        story.append(RLImage(tmp.name, width=15*cm, height=9*cm, kind="proportional"))
        story.append(Spacer(1, 0.5*cm))

    story.append(Paragraph(f"Rapport de contrôle qualité Cloud v5.0", sub_style))
    doc.build(story)
    
    pdf_bytes = pdf_buffer.getvalue()
    pdf_buffer.close()
    if "ann_img" in result:
        try: os.unlink(tmp.name)
        except: pass
    return pdf_bytes


# ─────────────────────────────────────────────
#  INTERFACE WEB COMPATIBLE SMARTPHONE/TABLETTE
# ─────────────────────────────────────────────

st.title("🔩 Vision Qualité Mobile")
st.write("Prenez une photo ou chargez un lot de pièces pour exécuter le calcul instantané.")

# Ajustement manuel de la sensibilité
sensibility = st.radio(
    "Configuration du lot :",
    ["low", "medium", "high"],
    index=1,
    format_func=lambda x: "Pièces très espacées" if x=="low" else "Lot Standard / Normal" if x=="medium" else "Pièces collées ou imbriquées"
)

# Gestion de la caméra sur Mobile
source_mode = st.selectbox("Méthode de capture :", ["📷 Prendre une Photo", "📂 Charger depuis la Galerie"])

img_file = None
if source_mode == "📷 Prendre une Photo":
    img_file = st.camera_input("Ouvrir l'appareil photo")
else:
    img_file = st.file_uploader("Sélectionner une image", type=["jpg", "jpeg", "png", "webp"])

if img_file is not None:
    # Lecture OpenCV de l'image reçue du smartphone
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Analyse
    count, annotated_cv = detect_and_count(img_cv, sensitivity)

    # Conversion d'affichage pour Streamlit
    ann_rgb = cv2.cvtColor(annotated_cv, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(ann_rgb)
    
    st.success(f"### 🎉 Analyse terminée : {count} éléments détectés")
    st.image(pil_img, caption="Résultat visuel de la détection", use_container_width=True)

    # Formulaire de traçabilité
    st.markdown("---")
    st.subheader("📋 Informations de traçabilité (Rapport)")
    op = st.text_input("Nom de l'opérateur")
    lot = st.text_input("Référence du lot")
    ptype = st.text_input("Désignation de la pièce")
    expected = st.text_input("Quantité théorique attendue")

    # Données compilées pour le PDF
    result_data = {
        "count": count, "operator": op, "lot_ref": lot,
        "piece_type": ptype, "expected": expected, "ann_img": pil_img
    }
    
    pdf_data = make_pdf_bytes(result_data)
    
    # Téléchargement direct sur la tablette
    st.download_button(
        label="📄 Télécharger le Rapport PDF de Contrôle",
        data=pdf_data,
        file_name=f"Rapport_Mobile_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        mime="application/pdf",
        use_container_width=True
    )
