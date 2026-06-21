#!/usr/bin/env python3
"""
Application Web de comptage de visserie / écrous
Optimisée pour Tablettes, Smartphones et Streamlit Cloud
Moteur : FastSAM + heuristiques de fusion adaptées aux vis / écrous
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
from ultralytics import FastSAM

st.set_page_config(
    page_title="Contrôle Qualité - Vision Industrielle",
    page_icon="🔩",
    layout="centered"
)


@st.cache_resource
def load_fastsam_model():
    """Charge FastSAM une seule fois et télécharge les poids si besoin."""
    return FastSAM("FastSAM-s.pt")


def detect_and_count(image_cv, sensitivity: str = "medium"):
    """
    Détection class-agnostic avec FastSAM.
    - Fusionne les morceaux d'un même screw quand FastSAM coupe la tête et la tige.
    - Filtre les petits masques de texte / watermark / bruit.
    """
    conf_thresh = {"low": 0.55, "medium": 0.40, "high": 0.25}
    iou_thresh = {"low": 0.80, "medium": 0.70, "high": 0.60}
    min_area = {"low": 700, "medium": 250, "high": 80}
    border_margin = {"low": 0.08, "medium": 0.06, "high": 0.04}

    img_h, img_w = image_cv.shape[:2]
    img_area = float(img_h * img_w)
    border_px = int(min(img_h, img_w) * border_margin[sensitivity])

    def axis_info(mask):
        ys, xs = np.where(mask)
        if xs.size < 5 or ys.size < 5:
            if xs.size == 0 or ys.size == 0:
                return {
                    "center": (0.0, 0.0),
                    "angle": 0.0,
                    "major": 1.0,
                    "minor": 1.0,
                    "elongation": 1.0,
                }
            x1, y1, x2, y2 = float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())
            major = max(x2 - x1 + 1.0, y2 - y1 + 1.0)
            minor = max(min(x2 - x1 + 1.0, y2 - y1 + 1.0), 1.0)
            return {
                "center": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                "angle": 0.0,
                "major": major,
                "minor": minor,
                "elongation": major / minor,
            }

        pts = np.column_stack((xs, ys)).astype(np.float32)
        mean = np.mean(pts, axis=0)
        centered = pts - mean
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, order]
        major_vec = eigvecs[:, 0]
        angle = float(np.degrees(np.arctan2(major_vec[1], major_vec[0])))
        proj_major = centered @ major_vec
        minor_vec = eigvecs[:, 1]
        proj_minor = centered @ minor_vec
        major = float(max(proj_major.max() - proj_major.min(), 1.0))
        minor = float(max(proj_minor.max() - proj_minor.min(), 1.0))
        return {
            "center": (float(mean[0]), float(mean[1])),
            "angle": angle,
            "major": major,
            "minor": minor,
            "elongation": major / max(minor, 1.0),
        }

    def bbox_iou(box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        x_left = max(ax1, bx1)
        y_top = max(ay1, by1)
        x_right = min(ax2, bx2)
        y_bottom = min(ay2, by2)
        if x_right <= x_left or y_bottom <= y_top:
            return 0.0
        inter = float((x_right - x_left) * (y_bottom - y_top))
        area_a = float((ax2 - ax1) * (ay2 - ay1))
        area_b = float((bx2 - bx1) * (by2 - by1))
        return inter / max(area_a + area_b - inter, 1.0)

    def norm_angle_diff(a, b):
        diff = abs(a - b) % 180.0
        return min(diff, 180.0 - diff)

    def point_line_distance(point, line_center, line_angle_deg):
        px, py = point
        cx, cy = line_center
        theta = np.deg2rad(line_angle_deg)
        ux, uy = np.cos(theta), np.sin(theta)
        dx, dy = px - cx, py - cy
        proj = dx * ux + dy * uy
        perp_x = dx - proj * ux
        perp_y = dy - proj * uy
        return float(np.hypot(perp_x, perp_y)), float(proj)

    def should_merge(a, b):
        if bbox_iou(a["bbox"], b["bbox"]) > 0.12:
            return True

        dist = float(np.hypot(a["center"][0] - b["center"][0], a["center"][1] - b["center"][1]))
        angle_diff = norm_angle_diff(a["angle"], b["angle"])
        axis_ref, other = (a, b) if a["elongation"] >= b["elongation"] else (b, a)

        if dist > max(axis_ref["major"], other["major"]) * 0.95:
            return False

        # Merge screw head + shaft pieces: one part is elongated and the other is compact.
        if axis_ref["elongation"] >= 2.0 and other["elongation"] <= 2.3 and angle_diff <= 22.0:
            perp_dist, proj = point_line_distance(other["center"], axis_ref["center"], axis_ref["angle"])
            axis_half = max(axis_ref["major"] / 2.0, 1.0)
            near_axis_end = abs(proj) >= axis_half * 0.30
            far_from_mid = abs(proj) <= axis_half * 1.15
            if perp_dist <= max(axis_ref["minor"] * 0.6, 10.0) and near_axis_end and far_from_mid:
                return True

        # Merge two elongated fragments when FastSAM cuts the same screw into multiple pieces.
        if a["elongation"] >= 2.0 and b["elongation"] >= 2.0 and angle_diff <= 12.0:
            perp_dist, proj = point_line_distance(b["center"], a["center"], a["angle"])
            gap_limit = max(min(a["major"], b["major"]) * 0.30, 12.0)
            if perp_dist <= max(min(a["minor"], b["minor"]) * 0.55, 8.0) and abs(proj) <= gap_limit:
                return True

        return False

    def fill_holes(mask):
        mask_u8 = mask.astype(np.uint8) * 255
        h, w = mask_u8.shape
        flood = mask_u8.copy()
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(flood, flood_mask, (0, 0), 255)
        flood_inv = cv2.bitwise_not(flood)
        filled = cv2.bitwise_or(mask_u8, flood_inv)
        return filled > 0

    def split_touching_mask(mask):
        """Split a filled mask into separate objects when touching parts are detected."""
        mask_u8 = (mask.astype(np.uint8) * 255)
        area = int(mask_u8.sum() / 255)
        if area == 0:
            return []

        ys, xs = np.where(mask)
        if xs.size == 0 or ys.size == 0:
            return [mask]

        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        bbox_w = max(x2 - x1 + 1, 1)
        bbox_h = max(y2 - y1 + 1, 1)
        bbox_area = float(bbox_w * bbox_h)
        fill_ratio = float(area) / max(bbox_area, 1.0)

        # Compact masks are usually nuts; do not split them further.
        if fill_ratio > 0.38 and area < img_area * 0.12:
            return [mask]

        props = axis_info(mask)
        if props["elongation"] < 1.9 and area < img_area * 0.10:
            return [mask]

        dist = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
        if dist.max() <= 0:
            return [mask]

        # Conservative peak threshold: only split if there are clearly separated body centers.
        peak_threshold = 0.50 if props["elongation"] >= 2.8 else 0.44 if props["elongation"] >= 2.2 else 0.38
        sure_fg = np.uint8(dist > (peak_threshold * dist.max())) * 255

        # Remove tiny peaks so holes / text don't become objects.
        peak_countours, _ = cv2.findContours(sure_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cleaned_peaks = np.zeros_like(sure_fg)
        for contour in peak_countours:
            if cv2.contourArea(contour) >= max(area * 0.04, 20):
                cv2.drawContours(cleaned_peaks, [contour], -1, 255, -1)
        sure_fg = cleaned_peaks

        num_labels, markers = cv2.connectedComponents(sure_fg)
        if num_labels <= 2:
            return [mask]

        # If only one strong peak exists, keep the object as a single instance.
        if num_labels == 2 and props["elongation"] < 2.2:
            return [mask]

        kernel = np.ones((3, 3), np.uint8)
        sure_bg = cv2.dilate(mask_u8, kernel, iterations=2)
        unknown = cv2.subtract(sure_bg, sure_fg)

        markers = markers + 1
        markers[unknown == 255] = 0

        marker_input = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(marker_input, markers)

        pieces = []
        for marker_id in np.unique(markers):
            if marker_id <= 1:
                continue
            piece = np.uint8(markers == marker_id)
            piece_area = int(piece.sum())
            if piece_area < min_area[sensitivity]:
                continue

            # Keep the split only if it is a plausible body fragment.
            py, px = np.where(piece)
            if px.size == 0 or py.size == 0:
                continue
            px1, py1, px2, py2 = int(px.min()), int(py.min()), int(px.max()), int(py.max())
            pw = max(px2 - px1 + 1, 1)
            ph = max(py2 - py1 + 1, 1)
            piece_fill = float(piece_area) / max(float(pw * ph), 1.0)
            if piece_fill < 0.08:
                continue
            pieces.append(piece.astype(bool))

        return pieces if pieces else [mask]

    def merge_union_masks(items):
        groups = []
        for item in items:
            placed = False
            for group in groups:
                if any(should_merge(item, other) for other in group):
                    group.append(item)
                    placed = True
                    break
            if not placed:
                groups.append([item])

        merged = []
        for group in groups:
            union_mask = np.zeros((img_h, img_w), dtype=bool)
            for item in group:
                union_mask |= item["mask"]

            union_mask = fill_holes(union_mask)

            ys, xs = np.where(union_mask)
            if xs.size == 0 or ys.size == 0:
                continue

            x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            area = int(union_mask.sum())
            if area < min_area[sensitivity]:
                continue

            bbox_area = float((x2 - x1 + 1) * (y2 - y1 + 1))
            fill_ratio = float(area) / max(bbox_area, 1.0)
            contours, _ = cv2.findContours(union_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            solidity = 1.0
            if contours:
                contour = max(contours, key=cv2.contourArea)
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                contour_area = cv2.contourArea(contour)
                solidity = float(contour_area) / max(hull_area, 1.0)

            merged.append({
                "mask": union_mask,
                "bbox": (x1, y1, x2, y2),
                "area": area,
                "score": max(item["score"] for item in group),
                "fill_ratio": fill_ratio,
                "solidity": solidity,
            })

        return merged

    try:
        model = load_fastsam_model()
        results = model.predict(
            source=image_cv,
            stream=False,
            conf=conf_thresh[sensitivity],
            iou=iou_thresh[sensitivity],
            imgsz=1024,
            retina_masks=True,
            verbose=False,
        )

        if not results:
            raise RuntimeError("FastSAM n'a renvoyé aucun résultat.")

        result = results[0]
        if result.masks is None or result.masks.data is None or len(result.masks.data) == 0:
            raise RuntimeError("FastSAM n'a renvoyé aucun masque.")

        masks = result.masks.data.detach().cpu().numpy().astype(bool)
        scores = np.ones(len(masks), dtype=float)
        if getattr(result, "boxes", None) is not None and getattr(result.boxes, "conf", None) is not None:
            scores = result.boxes.conf.detach().cpu().numpy()

        candidates = []
        for idx, mask in enumerate(masks):
            area = int(mask.sum())
            if area < min_area[sensitivity] or area > img_area * 0.70:
                continue

            ys, xs = np.where(mask)
            if xs.size == 0 or ys.size == 0:
                continue

            x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            bbox_w = max(x2 - x1 + 1, 1)
            bbox_h = max(y2 - y1 + 1, 1)
            bbox_area = float(bbox_w * bbox_h)
            fill_ratio = float(area) / bbox_area

            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            solidity = 0.0
            if contours:
                contour = max(contours, key=cv2.contourArea)
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                contour_area = cv2.contourArea(contour)
                solidity = float(contour_area) / max(hull_area, 1.0)

            touches_border = (
                x1 <= border_px or y1 <= border_px or
                x2 >= img_w - border_px - 1 or y2 >= img_h - border_px - 1
            )

            # Reject watermark / text-like fragments.
            if area < img_area * 0.015 and (fill_ratio < 0.20 or solidity < 0.25):
                continue
            if touches_border and area < img_area * 0.08:
                continue
            if fill_ratio < 0.10 and area < img_area * 0.04:
                continue

            mask = fill_holes(mask)
            props = axis_info(mask)
            candidates.append({
                "mask": mask,
                "bbox": (x1, y1, x2, y2),
                "area": area,
                "bbox_area": bbox_area,
                "fill_ratio": fill_ratio,
                "solidity": solidity,
                "score": float(scores[idx]) if idx < len(scores) else 1.0,
                "center": props["center"],
                "angle": props["angle"],
                "major": props["major"],
                "minor": props["minor"],
                "elongation": props["elongation"],
            })

        pruned_candidates = []
        for cand in candidates:
            contained = False
            for other in candidates:
                if cand is other:
                    continue
                overlap = np.logical_and(cand["mask"], other["mask"]).sum()
                if overlap == 0:
                    continue
                overlap_ratio = float(overlap) / max(float(cand["mask"].sum()), 1.0)
                if overlap_ratio >= 0.80 and cand["area"] <= other["area"] * 0.80:
                    contained = True
                    break
            if not contained:
                pruned_candidates.append(cand)

        pruned_candidates.sort(key=lambda item: (item["score"], item["area"]), reverse=True)
        final_items = []
        for item in merge_union_masks(pruned_candidates):
            for piece in split_touching_mask(fill_holes(item["mask"])):
                ys, xs = np.where(piece)
                if xs.size == 0 or ys.size == 0:
                    continue
                x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
                area = int(piece.sum())
                if area < min_area[sensitivity]:
                    continue
                bbox_area = float((x2 - x1 + 1) * (y2 - y1 + 1))
                fill_ratio = float(area) / max(bbox_area, 1.0)
                contours, _ = cv2.findContours(piece.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                solidity = 1.0
                if contours:
                    contour = max(contours, key=cv2.contourArea)
                    hull = cv2.convexHull(contour)
                    hull_area = cv2.contourArea(hull)
                    contour_area = cv2.contourArea(contour)
                    solidity = float(contour_area) / max(hull_area, 1.0)
                final_items.append({
                    "mask": piece,
                    "bbox": (x1, y1, x2, y2),
                    "area": area,
                    "score": item["score"],
                    "fill_ratio": fill_ratio,
                    "solidity": solidity,
                })

        final_items.sort(key=lambda item: (item["score"], item["area"]), reverse=True)

        annotated = image_cv.copy()
        for idx, cand in enumerate(final_items, 1):
            x1, y1, x2, y2 = cand["bbox"]
            contours, _ = cv2.findContours(cand["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cv2.drawContours(annotated, contours, -1, (0, 210, 50), 2)
            else:
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 210, 50), 2)
            cv2.putText(annotated, f"#{idx}", (x1 + 4, max(18, y1 + 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 210, 50), 2)

        count = len(final_items)
        cv2.rectangle(annotated, (5, 5), (220, 50), (0, 0, 0), -1)
        cv2.putText(annotated, f"Total: {count}", (12, 38),
                    cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 255, 150), 2)
        return count, annotated

    except Exception:
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 1.0)

        grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        magnitude = np.uint8(255 * magnitude / max(np.max(magnitude), 1))

        _, edge_mask = cv2.threshold(magnitude, 30, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        edge_mask = cv2.morphologyEx(edge_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        edge_mask = cv2.dilate(edge_mask, kernel, iterations=1)

        contours, _ = cv2.findContours(edge_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detected = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area[sensitivity] or area > img_area * 0.60:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if w < 5 or h < 5 or w > img_w * 0.95 or h > img_h * 0.95:
                continue
            detected.append((x, y, w, h, area))

        annotated = image_cv.copy()
        for idx, (x, y, w, h, _) in enumerate(detected, 1):
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 210, 50), 2)
            cv2.putText(annotated, f"#{idx}", (x + 4, y + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 210, 50), 2)

        count = len(detected)
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
    if ecart:
        v_data.append([ecart])
    vt = Table(v_data, colWidths=[17*cm])
    vt_styles = [
        ("BACKGROUND", (0,0), (-1,0), v_color),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("PADDING", (0,0), (-1,-1), 8),
    ]
    if ecart:
        vt_styles += [
            ("BACKGROUND", (0,1), (-1,1), colors.HexColor("#fdecea")),
            ("TEXTCOLOR", (0,1), (-1,1), colors.HexColor("#c0392b")),
            ("FONTNAME", (0,1), (-1,1), "Helvetica-Bold"),
        ]
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

    story.append(Paragraph("Rapport de contrôle qualité Cloud v5.0", sub_style))
    doc.build(story)

    pdf_bytes = pdf_buffer.getvalue()
    pdf_buffer.close()
    if "ann_img" in result:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass
    return pdf_bytes


# ─────────────────────────────────────────────
#  INTERFACE WEB COMPATIBLE SMARTPHONE/TABLETTE
# ─────────────────────────────────────────────

st.title("🔩 Vision Qualité Mobile")
st.write("Prenez une photo ou chargez un lot de pièces pour exécuter le calcul instantané.")

sensitivity = st.radio(
    "Configuration du lot :",
    ["low", "medium", "high"],
    index=1,
    format_func=lambda x: "Pièces très espacées" if x == "low" else "Lot Standard / Normal" if x == "medium" else "Pièces collées ou imbriquées"
)

source_mode = st.selectbox("Méthode de capture :", ["📷 Prendre une Photo", "📂 Charger depuis la Galerie"])

img_file = None
if source_mode == "📷 Prendre une Photo":
    img_file = st.camera_input("Ouvrir l'appareil photo")
else:
    img_file = st.file_uploader("Sélectionner une image", type=["jpg", "jpeg", "png", "webp"])

if img_file is not None:
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    img_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    count, annotated_cv = detect_and_count(img_cv, sensitivity)

    ann_rgb = cv2.cvtColor(annotated_cv, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(ann_rgb)

    st.success(f"### 🎉 Analyse terminée : {count} éléments détectés")
    st.image(pil_img, caption="Résultat visuel de la détection", use_container_width=True)

   # Contrôleur (liste déroulante)
    op = st.selectbox(
        "Contrôleur",
        [
            "Mur qualité",
            "Magasinier"
        ]
    )

    # Référence du lot
    lot = st.text_input("Référence du lot")

    # Désignation de la pièce (liste déroulante)
    ptype = st.selectbox(
        "Désignation de la pièce",
        [
            "NSA5057C5"
        ]
    )

    # Quantité théorique attendue (liste déroulante)
    expected = st.selectbox(
        "Quantité théorique attendue",
        [
            "18"
        ]
    )

    result_data = {
        "count": count,
        "operator": op,
        "lot_ref": lot,
        "piece_type": ptype,
        "expected": expected,
        "ann_img": pil_img,
    }

    pdf_data = make_pdf_bytes(result_data)

    st.download_button(
        label="📄 Télécharger le Rapport PDF de Contrôle",
        data=pdf_data,
        file_name=f"Rapport_Mobile_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )
