import os

import tensorflow as tf
import cv2
import numpy as np

def fusion_img(img1,img2):
    '''
    :param img1:背景图片
    :param img2:要嵌入的数字 原始的28*28
    :return:融合后的图片,以及对应的标签
    标签长度11，0代表没有数字，1-9代表对应的数字，10代表0
    '''
    img2 = 255 - img2
    img2 = np.concatenate((img2.reshape(28, 28, 1), img2.reshape(28, 28, 1), img2.reshape(28, 28, 1)), 2)
    size = np.random.randint(-5, 5)
    img2 = cv2.resize(img2, (28 + size, 28 + size))

    add1 = np.random.randint(0, 90)
    add2 = np.random.randint(0, 90)

    rows, cols, channels = img2.shape
    roi = img1[add1:add1 + rows, add2:add2 + cols]

    img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 200, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    img1_bg = cv2.bitwise_and(roi, roi, mask=mask)
    img2_fg = cv2.bitwise_and(img2, img2, mask=mask_inv)

    dst = cv2.add(img1_bg, img2_fg)
    img1[add1:add1 + rows, add2:add2 + cols] = dst
    return img1

def fusion_4img(bg,a,b,c,d):
    def fusion(img1, img2,add):
        '''
        :param img1:
        :param img2:
        :param add:1234四象限
        :return:
        '''
        img2 = 255 - img2
        # img2 = tf.concat((tf.reshape(img2,(28,28,1)),tf.reshape(img2,(28,28,1)),tf.reshape(img2,(28,28,1))),2)
        img2 = np.concatenate((img2.reshape(28, 28, 1), img2.reshape(28, 28, 1), img2.reshape(28, 28, 1)), 2)
        size = np.random.randint(-5, 5)
        img2 = cv2.resize(img2, (28 + size, 28 + size))

        if add==1:
            add1 = np.random.randint(0, 30)
            add2 = np.random.randint(0, 30)
        elif add==2:
            add1 = np.random.randint(50, 95)
            add2 = np.random.randint(0, 30)
        elif add==3:
            add1 = np.random.randint(0, 30)
            add2 = np.random.randint(50, 95)
        elif add==4:
            add1 = np.random.randint(50, 95)
            add2 = np.random.randint(50, 95)

        # print(leftuppoint)
        rows, cols, channels = img2.shape
        roi = img1[add1:add1 + rows, add2:add2 + cols]

        img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 200, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        img1_bg = cv2.bitwise_and(roi, roi, mask=mask)
        img2_fg = cv2.bitwise_and(img2, img2, mask=mask_inv)

        dst = cv2.add(img1_bg, img2_fg)
        img1[add1:add1 + rows, add2:add2 + cols] = dst
        return img1,add1,add2,size

    def precess_label(label,add1,add2,size,l):
        #打上标签
        x1 = int(add1 / 4)
        y1 = int(add2 / 4)
        x2 = int((add1 + 28 + size) / 4)
        y2 = int((add2 + 28 + size) / 4)
        for i in range(32):
            if i > y1 and i < y2:
                for j in range(32):
                    if j > x1 - 1 and j < x2 - 1:
                        label[j][i][10] = 0
                        label[j][i][l] = 1
        return label

    label = np.zeros((32,32,11))
    label[:, :, 10:11] = 1

    img1 = bg
    (img2,l1),(img3,l2),(img4,l3),(img5,l4) = a,b,c,d

    img,add1,add2,size = fusion(img1,img2,1)
    label = precess_label(label,add1,add2,size,l1)

    img,add1,add2,size = fusion(img,img3,2)
    label = precess_label(label, add1, add2, size, l2)

    img,add1,add2,size = fusion(img, img4, 3)
    label = precess_label(label, add1, add2, size, l3)

    img,add1,add2,size = fusion(img, img5, 4)
    label = precess_label(label, add1, add2, size, l4)

    return img,label

if __name__ == '__main__':

    (x, y), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    list = os.listdir('./train/bg')
    for i in range(5):
        bgi = np.random.randint(0, 36)
        bg = cv2.imread('./train/bg/%s'%list[bgi])
        img1 = np.random.randint(0,60000)
        img2 = np.random.randint(0, 60000)
        img3 = np.random.randint(0, 60000)
        img4 = np.random.randint(0, 60000)
        a = (x[img1],y[img1])
        b = (x[img2],y[img2])
        c = (x[img3],y[img3])
        d = (x[img4],y[img4])
        img, label = fusion_4img(bg,a,b,c,d)
        cv2.imwrite('./val/%s.jpg'%i,img)
        # np.save('./train/label/%s.npz'%i,label)
        # print(label.shape)



    # out = label[:, :, 0:10]
    # for i in range(32):
    #     for j in range(32):
    #         for k in range(10):
    #             if out[i][j][k] > 0:
    #                 # out[i][j][0] = 1
    #                 cv2.rectangle(img, (j * 4, i * 4), (j * 4 + 4, i * 4 + 4), (0, 0, 255), 1)
    #             else:
    #                 out[i][j][k] = 0
    #
    # cv2.imshow('res', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


