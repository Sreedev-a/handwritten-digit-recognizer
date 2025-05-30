from pathlib import Path
import pandas as pd
from PIL import Image
import streamlit as st
import torch
from streamlit_drawable_canvas import st_canvas
from model import DigitNet,predict

st.set_page_config(page_title="InkSense",page_icon="✎",layout="wide")
st.markdown("<style>.stApp{background:radial-gradient(circle at 70% 0,#2b2554,#080a11 42%)}.block-container{max-width:1000px;padding-top:3rem}[data-testid=stMetric]{background:#121426;border:1px solid #302e53;border-radius:18px;padding:16px}</style>",unsafe_allow_html=True)
st.title("InkSense"); st.caption("Draw a digit and inspect a CNN's complete MNIST probability distribution.")
path=Path("models/mnist_cnn.pt")
if not path.exists(): st.warning("Model weights are not present yet. Run `python train.py` once, then refresh."); st.stop()
@st.cache_resource
def load_model():
    model=DigitNet(); model.load_state_dict(torch.load(path,map_location="cpu",weights_only=True)); return model
canvas=st_canvas(fill_color="black",stroke_width=18,stroke_color="white",background_color="black",width=320,height=320,drawing_mode="freedraw",key="canvas")
if st.button("Recognize digit",type="primary"):
    try:
        digit,prob=predict(load_model(),Image.fromarray(canvas.image_data.astype("uint8"))); st.metric("Prediction",digit,f"{prob[digit]*100:.1f}% confidence"); st.bar_chart(pd.DataFrame({"digit":range(10),"probability":prob}).set_index("digit"))
    except ValueError as exc: st.error(str(exc))

