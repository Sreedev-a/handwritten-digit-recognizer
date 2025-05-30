import pytest,torch
from PIL import Image,ImageDraw
from model import DigitNet,preprocess,predict
def test_preprocess_shape():
    image=Image.new("L",(100,100)); ImageDraw.Draw(image).line((50,20,50,80),fill=255,width=12); assert preprocess(image).shape==(1,1,28,28)
def test_model_distribution():
    image=Image.new("L",(100,100)); ImageDraw.Draw(image).ellipse((20,10,80,90),outline=255,width=10); digit,p=predict(DigitNet(),image); assert digit in range(10); assert torch.isclose(torch.tensor(p.sum()),torch.tensor(1.0))
def test_empty():
    with pytest.raises(ValueError): preprocess(Image.new("L",(28,28)))

