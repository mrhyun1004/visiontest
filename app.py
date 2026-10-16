from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "mp_patch_detector.npz"

st.set_page_config(page_title="패치 메모리 이상 탐지", page_icon="🔎", layout="wide")
st.title("패치 메모리 이상 탐지")
st.caption("정상 이미지 패치 메모리와 비교해 이상 가능성을 확인합니다.")


@st.cache_resource
def load_detector(path: str, modified: float):
    del modified
    with np.load(path, allow_pickle=False) as saved:
        memory = saved["memory"].copy()
        image_size = tuple(int(v) for v in saved["image_size"])
        patch_size = int(saved["patch_size"])
        threshold = float(saved["threshold"])
    return NearestNeighbors(n_neighbors=1).fit(memory), image_size, patch_size, threshold


def score_image(image: Image.Image, nearest, image_size, patch_size):
    original = np.asarray(image.convert("RGB"))
    resized = cv2.resize(original, image_size, interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
    vectors = []
    for y in range(0, image_size[1], patch_size):
        for x in range(0, image_size[0], patch_size):
            patch = gray[y : y + patch_size, x : x + patch_size]
            edge = cv2.Canny(patch, 40, 120)
            vectors.append([patch.mean() / 255, patch.std() / 255, (edge > 0).mean()])
    distances, _ = nearest.kneighbors(np.asarray(vectors, dtype=np.float32))
    grid = distances.reshape(image_size[1] // patch_size, image_size[0] // patch_size)
    return original, float(grid.max()), grid


if not MODEL_PATH.is_file():
    st.error(f"모델 파일을 찾을 수 없습니다: {MODEL_PATH.name}")
    st.stop()

nearest, image_size, patch_size, threshold = load_detector(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)

uploaded = st.file_uploader("검사할 이미지를 올려 주세요", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"])
if uploaded is None:
    st.info("이미지를 올리면 이상 점수와 히트맵을 표시합니다.")
    st.stop()

try:
    image = Image.open(uploaded)
    original, score, grid = score_image(image, nearest, image_size, patch_size)
except Exception as exc:
    st.error(f"이미지를 처리하지 못했습니다: {exc}")
    st.stop()

is_anomaly = score >= threshold
label = "이상 가능성 있음" if is_anomaly else "정상 메모리와 유사"
col1, col2 = st.columns(2)
col1.metric("판정", label, delta="임계값 초과" if is_anomaly else "임계값 이하", delta_color="inverse" if is_anomaly else "normal")
col2.metric("이상 점수", f"{score:.4f}", f"임계값 {threshold:.4f}")

heat = cv2.resize(grid.astype(np.float32), (original.shape[1], original.shape[0]), interpolation=cv2.INTER_LINEAR)
heat_norm = cv2.normalize(heat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
heat_rgb = cv2.cvtColor(cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
overlay = cv2.addWeighted(original, 0.55, heat_rgb, 0.45, 0)
view1, view2 = st.columns(2)
view1.image(original, caption="원본 이미지", use_container_width=True)
view2.image(overlay, caption="이상 점수 히트맵 오버레이", use_container_width=True)
st.caption("히트맵은 패치 단위 점수를 확대해 표시합니다. 결과는 검토 신호이며 결함을 확정하지 않습니다.")
