'''
    实验测试代码
'''
import math
import cv2
import numpy as np
from index_count import getBoundingDiv, getAdivB, getRect, gray_matrix, getRGBHSV


def getAreaDivArc(cnt, Obinary=None):
    area = cv2.contourArea(cnt)
    pri = cv2.arcLength(cnt, True)
    div = area / pri  # 原始面积周长比
    roundness = (4 * np.pi * area) / (pri * pri)  # 圆形度
    lisan = pri * pri / area  # 离散度
    return div, roundness, lisan


# 自适应gamma校正
def adptive_gamma(image):
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean = np.mean(img_gray)
    fgamma = math.log10(0.5) / math.log10(mean / 255)  # 公式计算gamma
    image_gamma = np.uint8(np.power((np.array(image) / 255.0), fgamma) * 255.0)
    cv2.normalize(image_gamma, image_gamma, 0, 255, cv2.NORM_MINMAX)
    cv2.convertScaleAbs(image_gamma, image_gamma)
    return image_gamma


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


if __name__ == "__main__":

    for i in range(1, 13):  # 文件夹中一共12张图片
        print(i)
        img_path = "./images2/("+str(i)+").jpg"
        # img_path = "./darker22/("+str(i)+")_night.jpg"
        oringal_img = cv2.imread(img_path)
        oringal_img = adptive_gamma(oringal_img)  # 自适应gamma校正增强亮度
        Resized, edge, binary = preProcess(oringal_img)
        Resized_copy = np.copy(Resized)
        contours, hierarchy = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 获取多目标轮廓信息
        apple_num, orange_num, carambola_num, grape_num = 0, 0, 0, 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 100:  # 过滤小区域边界影响，获取精确目标
                continue
            else:
                objectType = "unknown"
                wdivh, x, y, k, h = getBoundingDiv(cnt)  # 获得目标的外接矩形坐标和长宽
                if k * h / 10 < 200:  # 过滤小区域误判影响，获取精确目标
                    continue
                ROI_All = cv2.bitwise_and(Resized_copy, Resized_copy, mask=binary)  # 获取全局ROI区域
                SingOJ = ROI_All[y:y+h, x:x+k, :]  # 获取局部ROI区域
                SingOb = binary[y:y+h, x:x+k]   # 获取局部ROI区域二值图像
                # cv2.imshow("demo", SingOJ)
                # cv2.waitKey(0)
                adivb = getAdivB(cnt)  # 获得似圆特征比
                adiva, roundness, lisan = getAreaDivArc(cnt, SingOb)  # 获得原始面积周长比、圆形度和离散度
                rectangularity, shenchang = getRect(cnt)  # 获得矩形度、伸长度
                Con, Eng, Asm, Idm, Auto_correlation = gray_matrix(ROI_All, x, y, k, h)  # 获得灰度共生矩阵得到的纹理特征
                RG, RB, GB = getRGBHSV(x, y, k, h, ROI_All)  # 获取|R-G| |R-B| |G-B|

                # 分类算法
                if adivb < 0.8:
                    if Auto_correlation <= 20 and Con > 1.25 and abs(RG-RB) <= 20:
                        objectType = "grape"
                        grape_num += 1
                    else:
                        objectType = "carambola"
                        carambola_num += 1
                else:
                    if abs(RG-RB) <= 10 and GB <= 10:
                        objectType = "apple"
                        apple_num += 1
                    else:
                        objectType = "orange"
                        orange_num += 1

                cv2.rectangle(Resized, (x, y), (x + k, y + h), (255, 0, 0)) # 绘制最小外接矩形
                cv2.putText(Resized, objectType, (x - 5, y - 5), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)

        # 显示识别结果
        text = f"apple:{apple_num}, orange:{orange_num}, carambola:{carambola_num}, grape:{grape_num}"
        cv2.putText(Resized, text, (5, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow("demo", Resized)
        cv2.waitKey(0)