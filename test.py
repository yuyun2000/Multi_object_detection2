import numpy
import tensorflow as tf
import cv2
import numpy as np

def onepoint(mat,x,y,k):
    #给一个01矩阵和一个值为1的点坐标，如果该点不孤立，返回点群的中心坐标,k是下标，也就是多分类的类别
    resultx = [x]
    resulty = [y]
    mat[x][y] = 0
    flag = 0 #判断孤立点
    for x1,y1 in [(x-1,y-1),(x,y-1),(x+1,y-1),(x-1,y),(x+1,y),(x-1,y+1),(x,y+1),(x+1,y+1)]:
        if mat[x1,y1] == 1:
            mat[x1, y1] = 0
            resultx.append(x1)
            resulty.append(y1)
            flag = 1
    for x2,y2 in [(x-2,y-2),(x-1,y-2),(x,y-2),(x+1,y-2),(x+2,y-2),(x-2,y-1),(x+2,y-1),(x-2,y),(x+2,y),(x-2,y+1),(x+2,y+1),(x-2,y+2),(x-1,y+2),(x,y+2),(x+1,y+2),(x+2,y+2)]:
        if mat[x2,y2] == 1:
            mat[x2, y2] = 0
            resultx.append(x2)
            resulty.append(y2)
            flag = 1
    if flag == 0:#两圈内没有其他点，认为是孤立点，舍弃
        return 0,0,k
    for x3,y3 in [(x-3,y-3),(x-2,y-3),(x-1,y-3),(x,y-3),(x+1,y-3),(x+2,y-3),(x+3,y-3),(x-3,y-2),(x+3,y-2),(x-3,y-1),(x+3,y-1),(x-3,y),(x+3,y),(x-3,y+1),(x+3,y+1),(x-3,y+2),(x+3,y+2),(x-3,y+3),(x-2,y+3),(x-1,y+3),(x,y+3),(x+1,y+3),(x+2,y+3),(x+3,y+3)]:
        if mat[x3,y3] == 1:
            mat[x3, y3] = 0
            resultx.append(x3)
            resulty.append(y3)
    for x4,y4 in [(x-4,y-4),(x-3,y-4),(x-2,y-4),(x-1,y-4),(x,y-4),(x+1,y-4),(x+2,y-4),(x+3,y-4),(x+4,y-4),(x-4,y-3),(x+4,y-3),(x-4,y-2),(x+4,y-2),(x-4,y-1),(x+4,y-1),(x-4,y),(x+4,y),(x-4,y+1),(x+4,y+1),(x-4,y+2),(x+4,y+2),(x-4,y+3),(x-4,y+3),(x-4,y+4),(x-3,y+4),(x-2,y+4),(x-1,y+4),(x,y+4),(x+1,y+4),(x+2,y+4),(x+3,y+4),(x+4,y+4)]:
        if mat[x4,y4] == 1:
            mat[x4, y4] = 0
            resultx.append(x4)
            resulty.append(y4)

    for x5,y5 in [(x-5,y-5),(x-4,y-5),(x-3,y-5),(x-2,y-5),(x-1,y-5),(x,y-5),(x+1,y-5),(x+2,y-5),(x+3,y-5),(x+4,y-5),(x+5,y-5),
                  (x-5,y-4),(x+5,y-4),(x-5,y-3),(x+5,y-3),(x-5,y-2),(x+5,y-2),(x-5,y-1),(x+5,y-1),
                  (x-5,y),(x+5,y),(x-5,y+1),(x+5,y+1),(x-5,y+2),(x+5,y+2),(x-5,y+3),(x-5,y+3),(x-5,y+4),(x-5,y+4),
                  (x-5,y+5),(x-4,y+5),(x-3,y+5),(x-2,y+5),(x-1,y+5),(x,y+5),(x+1,y+5),(x+2,y+5),(x+3,y+5),(x+4,y+5),(x+5,y+5)]:
        if mat[x5,y5] == 1:
            mat[x5, y5] = 0
            resultx.append(x5)
            resulty.append(y5)

    for x6,y6 in [(x-6,y-6),(x-5,y-6),(x-4,y-6),(x-3,y-6),(x-2,y-6),(x-1,y-6),(x,y-6),(x+1,y-6),(x+2,y-6),(x+3,y-6),(x+4,y-6),(x+5,y-6),(x+6,y-6),
                  (x-6,y-5),(x+6,y-4),(x-6,y-4),(x+6,y-4),(x-6,y-3),(x+6,y-3),(x-6,y-2),(x+6,y-2),(x-6,y-1),(x+6,y-1),
                  (x-6,y),(x+6,y),(x-6,y+1),(x+6,y+1),(x-6,y+2),(x+6,y+2),(x-6,y+3),(x-6,y+3),(x-6,y+4),(x-6,y+4),(x-6,y+5),(x-6,y+5),
                  (x-6,y+6),(x-5,y+6),(x-4,y+6),(x-3,y+6),(x-2,y+6),(x-1,y+6),(x,y+6),(x+1,y+6),(x+2,y+6),(x+3,y+6),(x+4,y+6),(x+5,y+6),(x+6,y+6)]:
        if mat[x6,y6] == 1:
            mat[x6, y6] = 0
            resultx.append(x6)
            resulty.append(y6)
    import numpy as np
    if len(resultx)<10:
        return 0,0,k
    return np.mean(resultx),numpy.mean((resulty)),k



def my_softmax(x):
    exp_x = np.exp(x)
    return exp_x/np.sum(exp_x)

model = tf.keras.models.load_model("./shuzi32.h5")


imagegt = cv2.imread('./val/1.jpg')
imagegt = cv2.resize(imagegt,(128,128))
image = imagegt.astype(np.float32)
img = image / 255
img = img.reshape(1,128,128,3)
out = model(img,training=True)
out = np.array(tf.reshape(out[0:1,:,:,:],(32,32,11)))
# for i in range(64):
    # for j in range(64):
        # out[i][j]=my_softmax(out[i][j])
print(out)

for i in range(32):
    for j in range(32):
        idx = np.argmax(out[i][j])
        if idx!=10:
            # if out[i][j][idx]>350:
                out[i][j][idx] = 1
                if idx <=3:
                    cv2.rectangle(imagegt,(j*4,i*4),(j*4+4,i*4+4),(255,0,0),1)
                elif idx <= 6:
                    cv2.rectangle(imagegt, (j * 4, i * 4), (j * 4 + 4, i * 4 + 4), (0, 255, 0), 1)
                else:
                    cv2.rectangle(imagegt, (j * 4, i * 4), (j * 4 + 4, i * 4 + 4), (0, 0, 255), 1)
            # else:
            #     out[i][j][idx] = 0

for i in range(6,25):
    for j in range(6,25):
        for k in range(10):
            if out[i][j][k] == 1:
                temp = out[:,:,k:k+1]
                x, y ,cls= onepoint(temp, i, j,k)
                if x != 0:
                    print(cls)
                    # cv2.circle(imagegt, (int(y * 4), int(x * 4)), 1, (0, 255, 255),thickness=1)
                    cv2.putText(imagegt,'%s'%k,(int(y * 4)-5, int(x * 4)+5),1,1,(0,0,255),1) #写字体的时候会偏移，手动矫正一下

imagegt = cv2.resize(imagegt,(320,320))
cv2.imshow('1',imagegt)
cv2.waitKey(0)
cv2.destroyAllWindows()

#---------------------------下面是视频测试------------------------------------------

# vid = cv2.VideoCapture('./val/t3.mp4')
# while True:
#     flag,img = vid.read(0)
#     if not flag:
#         break
#     img0 = cv2.resize(img,(128,128))
#     img = img0.astype(np.float32)
#     img = img / 255
#     img = img.reshape(1, 128, 128, 3)
#     out = model(img, training=True)
#
#     out = np.array(tf.reshape(out[0:1,:,:,:],(32,32,11)))
#     # for i in range(64):
#     #     for j in range(64):
#     #         out[i][j]=my_softmax(out[i][j])
#     # print(out)
#     for i in range(32):
#         for j in range(32):
#             idx = np.argmax(out[i][j])
#             if idx!=10:
#                 if out[i][j][idx]>500:
#                     out[i][j][idx] = 1
#                 else:
#                     out[i][j][idx] = 0
#
#     for i in range(6,25):
#         for j in range(6,25):
#             for k in range(10):
#                 if out[i][j][k] == 1:
#                     temp = out[:,:,k:k+1]
#                     x, y ,cls= onepoint(temp, i, j,k)
#                     if x != 0:
#                         print(cls)
#                         cv2.putText(img0,'%s'%k,(int(y * 4)-5, int(x * 4)+5),1,1,(0,0,255),1)
#
#     img0 = cv2.resize(img0, (256, 256))
#     cv2.imshow('1', img0)
#     if ord('q') == cv2.waitKey(1):
#         break
# vid.release()
# #销毁所有的数据
# cv2.destroyAllWindows()