from PIL import Image, ImageDraw, ImageFont
from random import randint
import numpy as np
import math
from glob import glob


FONTPATH = './fonts/' # couri calibri time 
CHARS = "23456789ABCDEFGHJKMNPQRSTUVWXYZ" # 無 0 O I L 1


class line:
    def __init__(self, Size, MinLength, MaxLength, Width):
        length = np.random.randint(MinLength, MaxLength)
        xStart = np.random.randint(1, Size[0])
        yStart = np.random.randint(1, Size[1])
        angle = randint(0, 360)
        radians = math.radians(angle)
        xEnd = xStart + round(math.sin(radians) * length)
        yEnd = yStart + round(math.cos(radians) * length)
        self.location = [xStart, yStart, xEnd, yEnd]
        self.color = (200, 200, 200)
        self.width = Width
    def draw(self, ImageDraw):
        ImageDraw.line(xy=self.location, fill=self.color, width=self.width)

class captchText:
    def __init__(self, trueType, char, xOffset, yOffset, imageSize, rotate=False):
        self.char = char
        self.color = (randint(30,125), randint(30,125), randint(30,125))
        self.angle = randint(-3, 3)
        self.xOffset = xOffset
        self.yOffset = yOffset
        self.rotate = rotate
        self.trueType = trueType
        self.imageSize = imageSize
    def draw(self, image):
        font = ImageFont.truetype(self.trueType, randint(18, 32))
        fontSize = font.getsize(self.char)
        text = Image.new("RGBA", (fontSize[0], fontSize[1]), (0, 0, 0, 0))
        textdraw = ImageDraw.Draw(text)
        textdraw.text((0, 0), self.char, font=font, fill=self.color)
        if self.rotate:
            text = text.rotate(self.angle, expand=True)
        image.paste(text, (self.xOffset, self.yOffset - fontSize[1]), text)
        


def addGasussNoise(image, mean=0.1, var=0.001):
    image = np.array(image/255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    image[:,:,0] = image[:,:,0] + noise[:,:,0]
    image[:,:,1] = image[:,:,1] + noise[:,:,0]
    image[:,:,2] = image[:,:,2] + noise[:,:,0]
    out = image
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out*255)
    return out

def generate(GenNum, SavePath, TextNum, LineNum, LineLength, CaptchaSize=(80,28)):
    for num in range(GenNum):
        # 建立空驗證碼圖
        Captcha = Image.new("RGB", CaptchaSize, (np.random.randint(210,255), np.random.randint(210,255), np.random.randint(210,255)))
        
        # 畫線 
        Lines = [line(CaptchaSize, LineLength[0], LineLength[1], 1) for _ in range(LineNum)]
        for Line in Lines:
            Line.draw(ImageDraw.Draw(Captcha))
        
        # 畫字 起點 寬6~10, 高22
        Answer = []
        FontTrueType =  glob(FONTPATH + "*")
        for i in range(TextNum):
            char = CHARS[randint(0, len(CHARS)-1)]
            Answer.append(char)
            captchText_ = captchText(FontTrueType[randint(0, len(FontTrueType)-1)], char, xOffset=randint(6,10)+13*i, yOffset=22, imageSize=CaptchaSize, rotate=False)
            captchText_.draw(Captcha)
        
        # 加高斯雜訊
        CaptchaArray = np.asarray(Captcha)
        CaptchaArray = addGasussNoise(CaptchaArray, 0, 0.0001)
        Captcha = Image.fromarray(CaptchaArray)

        Captcha.save(SavePath + "%s.png" % ''.join(Answer))



if __name__ == "__main__":
    generate(20, "./data/images_Aug/", TextNum=5, LineNum=100, LineLength=(15,25))
