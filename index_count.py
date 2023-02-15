'''
    指标计算代码
'''
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt


# 获取二值图像
def get_binary(image):
    blur = cv2.pyrMeanShiftFiltering(image, sp=10, sr=100)  # 边缘保留滤波EPF
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)  # 转成灰度图像
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)  # 自适应阈值
    return binary


# 数据预处理
def preProcess(Img):
    OriResize = cv2.resize(Img, (1200, 600))
    # cv2.imshow("demo", OriResize)
    # cv2.waitKey(0)
    get = OriResize.copy()
    mask = get_binary(get)
    blur = cv2.GaussianBlur(mask, (7, 7), 1)  # 高斯模糊
    canny = cv2.Canny(blur, 50, 50)  # canny算子提取边缘
    kenel = np.ones((3, 3))
    mask1 = cv2.dilate(canny, kenel, iterations=1)  # 膨胀
    mask2 = cv2.erode(mask1, kenel, iterations=1)  # 腐蚀
    return OriResize, mask2, blur


'''颜色特征'''
# 计算|R-G| |R-B| |G-B|
def getRGBHSV(x, y, k, h, Ryuanshi):
    B, G, R = cv2.split(Ryuanshi)
    fMB = 0
    fMG = 0
    fMR = 0
    Rcount = 0
    for i in range(x, x + k):
        for j in range(y, y + h):
            if B[j, i] == 0 and G[j, i] == 0 and R[j, i] == 0:
                continue
            else:
                fMB += B[j, i]
                fMG += G[j, i]
                fMR += R[j, i]
                Rcount += 1
    fMB /= Rcount
    fMG /= Rcount
    fMR /= Rcount

    return abs(fMR - fMG), abs(fMR - fMB), abs(fMG - fMB)


# 计算R均值 G均值 B均值 R+G R均方根 G均方根 B均方根
def getMRGBAndSqrt(x, y, k, h, Ryuanshi):
    B, G, R = cv2.split(Ryuanshi)
    fMB = 0
    fMG = 0
    fMR = 0
    Rcount = 0
    for i in range(x, x + k):
        for j in range(y, y + h):
            if B[j, i] == 0 and G[j, i] == 0 and R[j, i] == 0:
                continue
            else:
                fMB += B[j, i]
                fMG += G[j, i]
                fMR += R[j, i]
                Rcount += 1
    fMB /= Rcount
    fMG /= Rcount
    fMR /= Rcount

    # 求解方差
    fcB = 0
    fcG = 0
    fcR = 0
    for i in range(x, x + k):
        for j in range(y, y + h):
            if B[j, i] == 0 and G[j, i] == 0 and R[j, i] == 0:
                continue
            else:
                fcB += pow((B[j, i] - fMB), 2)
                fcG += pow((G[j, i] - fMG), 2)
                fcR += pow((R[j, i] - fMR), 2)
    fcB = math.sqrt(fcB / Rcount)
    fcG = math.sqrt(fcG / Rcount)
    fcR = math.sqrt(fcR / Rcount)

    return fMR, fMG, fMB, fcR, fcG, fcB, abs(fMR + fMG), abs(fMB - fMG)


# 计算H均值 S均值 V均值 平均HSV
def getHSV(x, y, k, h, Ryuanshi):
    HSV = cv2.cvtColor(Ryuanshi, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(HSV)
    fMH = 0
    fMS = 0
    fMV = 0
    Rcount = 0
    for i in range(x, x + k):
        for j in range(y, y + h):
            if 0 <= H[j, i] <= 180 and 0 <= S[j, i] <= 255 and 0 <= V[j, i] <= 46:
                continue
            else:
                fMH += H[j, i]
                fMS += S[j, i]
                fMV += V[j, i]
                Rcount += 1
    fMH /= Rcount
    fMS /= Rcount
    fMV /= Rcount
    return fMH, fMS, fMV, (fMH + fMS + fMV) / 3


'''形状特征'''
# 计算原始面积周长比 圆形度 离散度
def getAreaDivArc(cnt, Obinary=None):
    my_area = 0
    for i in range(Obinary.shape[0]):
        for j in range(Obinary.shape[1]):
            if Obinary[i][j] == 255:
                my_area += 1
    # area = cv2.contourArea(cnt)
    pri = cv2.arcLength(cnt, True)
    div = my_area / pri
    roundness = (4 * np.pi * my_area) / (pri * pri)
    lisan = pri * pri / my_area
    return div, roundness, lisan


# 计算最小外接矩形框的坐标和长宽
def getBoundingDiv(cnt):
    pri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * pri, True)
    x, y, k, h = cv2.boundingRect(approx)
    # 加入定位框比
    if k > h:
        return k / h, x, y, k, h
    else:
        return h / k, x, y, k, h


# 计算似圆特征比
def getAdivB(cnt):
    ((x, y), (a, b), angle) = cv2.fitEllipse(cnt)
    # cv2.ellipse(Ryuanshi, (int(x), int(y)), (int(a), int(b)), angle, 0, 360, color=(0, 0, 255), thickness=2)
    return a / b


# 计算矩形度 伸长度
def getRect(cnt):
    ((x, y), (width, height), angle) = cv2.minAreaRect(cnt)
    area = cv2.contourArea(cnt)
    minrectmianji = height * width
    rectangularity = area / minrectmianji
    if height > width:
        shenchang = height / width
    else:
        shenchang = width / height
    return rectangularity, shenchang


'''纹理特征'''
# 灰度共生矩阵https://blog.csdn.net/u013066730/article/details/109776522
gray_level = 16
def maxGrayLevel(img):
    max_gray_level = 0
    (height, width) = img.shape
    for y in range(height):
        for x in range(width):
            if img[y][x] > max_gray_level:
                max_gray_level = img[y][x]
    return max_gray_level + 1


def getGlcm(input, d_x, d_y):
    srcdata = input.copy()
    ret = [[0.0 for i in range(gray_level)] for j in range(gray_level)]
    (height, width) = input.shape

    max_gray_level = maxGrayLevel(input)
    # 若灰度级数大于gray_level，则将图像的灰度级缩小至gray_level，减小灰度共生矩阵的大小
    if max_gray_level > gray_level:
        for j in range(height):
            for i in range(width):
                srcdata[j][i] = srcdata[j][i] * gray_level / max_gray_level

    if d_x >= 0 or d_y >= 0:
        for j in range(height - d_y):
            for i in range(width - d_x):
                rows = srcdata[j][i]
                cols = srcdata[j + d_y][i + d_x]
                ret[rows][cols] += 1.0
    else:
        for j in range(height):
            for i in range(width):
                rows = srcdata[j][i]
                cols = srcdata[j + d_y][i + d_x]
                ret[rows][cols] += 1.0

    for i in range(gray_level):
        for j in range(gray_level):
            ret[i][j] /= float(height * width)

    return ret


def feature_computer(p):
    mean = 0.0  # mean:均值
    Con = 0.0  # Con:对比度
    Eng = 0.0  # Eng:熵
    Asm = 0.0  # Asm:角二阶矩（能量）
    Idm = 0.0  # Idm:反差分矩阵
    Auto_correlation = 0.0  # Auto_correlation：相关性
    std2 = 0.0
    std = 0.0
    for i in range(gray_level):
        for j in range(gray_level):
            mean += p[i][j] * i / gray_level ** 2
            Con += (i - j) * (i - j) * p[i][j]
            Asm += p[i][j] * p[i][j]
            Idm += p[i][j] / (1 + (i - j) * (i - j))
            Auto_correlation += p[i][j] * i * j
            if p[i][j] > 0.0:
                Eng += p[i][j] * math.log(p[i][j])
        for i in range(gray_level):
            for j in range(gray_level):
                std2 += (p[i][j] * i - mean) ** 2
        std = np.sqrt(std2)
    return Con, -Eng, Asm, Idm, Auto_correlation


def gray_matrix(img, x, y, k, h):
    img = img[y:y + h, x:x + k, :]
    img_shape = img.shape
    img = cv2.resize(img, (img_shape[1] // 2, img_shape[0] // 2), interpolation=cv2.INTER_CUBIC)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    glcm_0 = getGlcm(img_gray, 1, 0)  # 1,0 时为水平相邻：也就是0度时；
    Con, Eng, Asm, Idm, Auto_correlation = feature_computer(glcm_0)
    return Con, Eng, Asm, Idm, Auto_correlation


def submain(path):
    oringal_img = cv2.imread(path)
    Resized, edge, binary = preProcess(oringal_img)
    contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 获取多目标轮廓信息
    if path.__contains__("grape"):
        grape_contours = [np.concatenate((contours[13], contours[14], contours[15]), 0)]
        contours = grape_contours
    contours = contours[-1]
    adiva, roundness, lisan = getAreaDivArc(contours, binary)  # 获得原始面积周长比、圆形度和离散度
    wdivh, x, y, k, h = getBoundingDiv(contours)  # 获得目标的外接矩形坐标和长宽
    rectangularity, shenchang = getRect(contours)  # 获得矩形度、伸长度
    adivb = getAdivB(contours)  # 获得似圆特征比

    RG, RB, GB = getRGBHSV(x, y, k, h, binary)  # 获取|R-G| |R-B| |G-B|
    fMR, fMG, fMB, fcR, fcG, fcB, RAG, BSG = getMRGBAndSqrt(x, y, k, h, binary)  # 获取RGB相关信息
    fMH, fMS, fMV, MeanHSV = getHSV(x, y, k, h, binary)  # 获取HSV信息

    Con, Eng, Asm, Idm, Auto_correlation = gray_matrix(binary, x, y, k, h)  # 获得灰度共生矩阵得到的纹理特征

    print(f"颜色特征:\n R均值:{fMR}, G均值:{fMG}, B均值:{fMB}, R+G:{RAG}, R均方根:{fcR}, G均方根:{fcG}, B均方根:{fcB}\n "
          f"H均值:{fMH}, S均值:{fMS}, V均值:{fMV}, 平均HSV:{MeanHSV}, |R-G|:{RG}, |R-B|:{RB}, |G-B|:{GB}\n")

    print(f"RGB颜色特征:\n R-G:{RG}, R-B:{RB}, G-B:{GB}\n")

    print(f"形状特征:\n 原始面积周长比:{adiva}, 伸长度:{shenchang}, 似圆特征比:{adivb}\n "
          f"圆形度:{roundness}, 矩形度:{rectangularity}\n")

    print(f"纹理特征:\n 对比度:{Con}, 能量:{Asm}, 信息熵:{Eng}, 反差分矩阵:{Idm}, 相关性:{Auto_correlation}\n")

    color_features = [fMR, fMG, fMB, RAG, fcR, fcG, fcB, fMH, fMS, fMV, MeanHSV]
    rgb_color_features = [RG, RB, GB]
    shape_features = [adiva, adivb, roundness, rectangularity, shenchang]
    # shape_features = [adivb, roundness, rectangularity, shenchang]
    texture_features = [Con, Asm, Eng, Idm, Auto_correlation]
    # texture_features = [Con, Asm, Eng, Idm]
    return color_features, shape_features, texture_features, rgb_color_features


if __name__== "__main__" :
    rgbhsv_color_features_x = ["R-G", "R-B", "G-B"]
    color_features_x = ["R均值", "G均值", "B均值", "R+G", "R均方根", "G均方根", "B均方根", "H均值", "S均值", "V均值", "HSV均值"]
    shape_features_x = ["原始面积周长比", "似圆特征比", "圆形度", "矩形度", "伸长度"]
    # shape_features_x = ["似圆特征比", "圆形度", "矩形度", "伸长度"]
    texture_features_x = ["对比度", "能量", "信息熵", "反差分矩阵", "相关性"]
    # texture_features_x = ["对比度", "能量", "信息熵", "反差分矩阵"]
    apple_color_features, apple_shape_features, apple_texture_features, rgb_apple_color_features = submain("./single/apple.jpg")
    orange_color_features, orange_shape_features, orange_texture_features, rgb_orange_color_features = submain("./single/orange.jpg")
    yangtao_color_features, yangtao_shape_features, yangtao_texture_features, rgb_yangtao_color_features = submain("./single/yangtao.jpg")
    grape_color_features, grape_shape_features, grape_texture_features, rgb_grape_color_features = submain("./single/grape.jpg")

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题
    plt.rcParams['axes.unicode_minus'] = False  # 解决中文显示问题
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(color_features_x, apple_color_features, label='苹果')
    ax.plot(color_features_x, orange_color_features, label='橘子')
    ax.plot(color_features_x, yangtao_color_features, label='杨桃')
    ax.plot(color_features_x, grape_color_features, label='葡萄')
    leg = ax.legend()
    fig.suptitle("颜色特征", fontsize=16, x=0.5, y=0.95)
    plt.savefig('./颜色特征.png', bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(shape_features_x, apple_shape_features, label='苹果')
    ax.plot(shape_features_x, orange_shape_features, label='橘子')
    ax.plot(shape_features_x, yangtao_shape_features, label='杨桃')
    ax.plot(shape_features_x, grape_shape_features, label='葡萄')
    leg = ax.legend()
    fig.suptitle("形状特征", fontsize=16, x=0.5, y=0.95)
    plt.savefig('./形状特征.png', bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(texture_features_x, apple_texture_features, label='苹果')
    ax.plot(texture_features_x, orange_texture_features, label='橘子')
    ax.plot(texture_features_x, yangtao_texture_features, label='杨桃')
    ax.plot(texture_features_x, grape_texture_features, label='葡萄')
    leg = ax.legend()
    fig.suptitle("纹理特征", fontsize=16, x=0.5, y=0.95)
    plt.savefig('./纹理特征.png', bbox_inches='tight')

    plt.rcParams['font.sans-serif'] = ['SimHei'] #解决中文显示问题
    plt.rcParams['axes.unicode_minus'] = False #解决中文显示问题
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rgbhsv_color_features_x, rgb_apple_color_features, label='苹果')
    ax.plot(rgbhsv_color_features_x, rgb_orange_color_features, label='橘子')
    ax.plot(rgbhsv_color_features_x, rgb_yangtao_color_features, label='杨桃')
    ax.plot(rgbhsv_color_features_x, rgb_grape_color_features, label='葡萄')
    leg = ax.legend()
    fig.suptitle("RGB之间的特征", fontsize=16, x=0.5, y=0.95)
    plt.savefig('./颜色特征RGB.png', bbox_inches='tight')