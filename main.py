from flask import Flask, render_template, request
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import base64
import io
import torch
import torchvision.transforms as transforms

app = Flask(__name__)

# 加载预训练的ResNet模型
model = torch.load("resnet18.pt")
model.eval()




# 图像预处理

# print(model)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/recognize', methods=['POST'])
def recognize():
    img_data = request.form.get('image')  # 获取前端传回的图像数据
    img_data = img_data.split(',')[-1]  # 去除Base64编码前缀

    # 解码Base64字符串并将其转换为图像对象
    img_bytes = base64.b64decode(img_data)
    img = Image.open(io.BytesIO(img_bytes))
    rgb_image = Image.new("RGB", img.size, (255, 255, 255))
    rgb_image.paste(img, mask=img.split()[3])
    img = rgb_image
    plt.imshow(rgb_image)
    plt.show()
    img = img.convert('L')
    inverted_image = Image.new("L", img.size)
    pixels = img.load()
    inverted_pixels = inverted_image.load()

    for i in range(img.width):
        for j in range(img.height):
            inverted_pixels[i, j] = 255 - pixels[i, j]
            print(inverted_pixels[i, j])
    plt.imshow(inverted_image)
    plt.show()
    img= inverted_image
    transform = transforms.Compose([transforms.Resize(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(0.5, 0.5)])
    with torch.no_grad():
        img = transform(img)
        img = img.cuda().unsqueeze(0)
        print("shape", img.shape)
        out = model(img)
        a = torch.softmax(out, 1)
        prb, predict = torch.max(a, 1)
        print(prb, predict.item())
    # 图像预处理

    # 使用预训练的模型进行预测

    # 返回识别结果给前端
    return "预测结果为"+str(predict.item())+"概率为"+str(prb.item())


if __name__ == '__main__':
    app.run(debug=True)
