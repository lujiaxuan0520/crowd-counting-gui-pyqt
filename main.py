from sys import argv,exit
from cv2 import imread,imwrite
from os import listdir
from os.path import splitext
import mcnn
import retinanet
import yolo
#from PyQt4 import QtCore, QtGui, uic
from PyQt4 import QtGui, uic

qtCreatorFile = "layout.ui"  # Enter file here.
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)
data_path_base = './data/'
category=['restaurant','outdoor','classroom']
p=0.5#RetinaNet所占权重
q=0.3#MCNN所占权重
r=0.2 #YOLO所占权重

class MyApp(QtGui.QMainWindow, Ui_MainWindow):
    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        #点击“加载图片”后，执行loadImage函数
        self.show_image_button.clicked.connect(self.loadImage)
        #点击“计算人数”后，执行calculate_num函数
        self.calculate_num_button.clicked.connect(self.calculate_num)
        #点击“清空”后，执行clear函数
        self.clear_button.clicked.connect(self.clear)

    #清空
    def clear(self):
        self.clear_num()
        self.clear_image()

    #加载图片
    def loadImage(self):
        mcnn.reloadData()
        #self.image_1.setPixmap(data_loader.blob_list[0]['data'])
        self.changeJPGtoPNG()#将JPG转换为PNG

        self.pix = QtGui.QPixmap(data_path_base+category[0]+'/1.png') #QPixmap不可读入jpg
        self.res_image_1.setPixmap(self.pix)
        self.res_image_1.setScaledContents(True) #使图像匹配label大小
        self.pix = QtGui.QPixmap(data_path_base + category[0] + '/2.png')  # QPixmap不可读入jpg
        self.res_image_2.setPixmap(self.pix)
        self.res_image_2.setScaledContents(True)  # 使图像匹配label大小
        self.pix = QtGui.QPixmap(data_path_base + category[0] + '/3.png')  # QPixmap不可读入jpg
        self.res_image_3.setPixmap(self.pix)
        self.res_image_3.setScaledContents(True)  # 使图像匹配label大小
        self.pix = QtGui.QPixmap(data_path_base + category[0] + '/4.png')  # QPixmap不可读入jpg
        self.res_image_4.setPixmap(self.pix)
        self.res_image_4.setScaledContents(True)  # 使图像匹配label大小
        self.pix = QtGui.QPixmap(data_path_base + category[0] + '/5.png')  # QPixmap不可读入jpg
        self.res_image_5.setPixmap(self.pix)
        self.res_image_5.setScaledContents(True)  # 使图像匹配label大小
        self.pix = QtGui.QPixmap(data_path_base + category[0] + '/6.png')  # QPixmap不可读入jpg
        self.res_image_6.setPixmap(self.pix)
        self.res_image_6.setScaledContents(True)  # 使图像匹配label大小

        self.pix = QtGui.QPixmap(data_path_base + category[1] + '/1.png')  # QPixmap不可读入jpg
        self.out_image_1.setPixmap(self.pix)
        self.out_image_1.setScaledContents(True)  # 使图像匹配label大小
        self.pix = QtGui.QPixmap(data_path_base + category[1] + '/2.png')  # QPixmap不可读入jpg
        self.out_image_2.setPixmap(self.pix)
        self.out_image_2.setScaledContents(True)  # 使图像匹配label大小
        self.pix = QtGui.QPixmap(data_path_base + category[1] + '/3.png')  # QPixmap不可读入jpg
        self.out_image_3.setPixmap(self.pix)
        self.out_image_3.setScaledContents(True)  # 使图像匹配label大小

        self.pix = QtGui.QPixmap(data_path_base + category[2] + '/1.png')  # QPixmap不可读入jpg
        self.class_image_1.setPixmap(self.pix)
        self.class_image_1.setScaledContents(True)  # 使图像匹配label大小
        self.pix = QtGui.QPixmap(data_path_base + category[2] + '/2.png')  # QPixmap不可读入jpg
        self.class_image_2.setPixmap(self.pix)
        self.class_image_2.setScaledContents(True)  # 使图像匹配label大小
        self.pix = QtGui.QPixmap(data_path_base + category[2] + '/3.png')  # QPixmap不可读入jpg
        self.class_image_3.setPixmap(self.pix)
        self.class_image_3.setScaledContents(True)  # 使图像匹配label大小
        self.pix = QtGui.QPixmap(data_path_base + category[2] + '/4.png')  # QPixmap不可读入jpg
        self.class_image_4.setPixmap(self.pix)
        self.class_image_4.setScaledContents(True)  # 使图像匹配label大小
        self.pix = QtGui.QPixmap(data_path_base + category[2] + '/5.png')  # QPixmap不可读入jpg
        self.class_image_5.setPixmap(self.pix)
        self.class_image_5.setScaledContents(True)  # 使图像匹配label大小
        self.pix = QtGui.QPixmap(data_path_base + category[2] + '/6.png')  # QPixmap不可读入jpg
        self.class_image_6.setPixmap(self.pix)
        self.class_image_6.setScaledContents(True)  # 使图像匹配label大小

    #将data_path文件夹下的JPG转为PNG
    def changeJPGtoPNG(self):
        for i in category:
            path=data_path_base+i+'/'
            for filename in listdir(path):
                if splitext(filename)[1] == '.jpg':
                    img = imread(path + filename)
                    newfilename = filename.replace(".jpg", ".png")
                    imwrite(path + newfilename, img)

    #计算人数
    def calculate_num(self):
        num_mcnn=mcnn.returnNum("restaurant",1)# 用mcnn获取第1张图片的总人数
        num_retinanet=retinanet.returnNum("restaurant",1)#用RetinaNet获取第1张图片的总人数
        num_yolo=yolo.returnNum("restaurant",1)
        num=p*num_retinanet+q*num_mcnn+r*num_yolo
        self.res_result_1.setText(str(format(num,".2f")))
        num_mcnn = mcnn.returnNum("restaurant", 2)  # 用mcnn获取第1张图片的总人数
        num_retinanet = retinanet.returnNum("restaurant", 2)  # 用RetinaNet获取第1张图片的总人数
        num_yolo = yolo.returnNum("restaurant", 2)
        num = p * num_retinanet + q * num_mcnn + r * num_yolo
        self.res_result_2.setText(str(format(num, ".2f")))
        num_mcnn = mcnn.returnNum("restaurant", 3)  # 用mcnn获取第1张图片的总人数
        num_retinanet = retinanet.returnNum("restaurant", 3)  # 用RetinaNet获取第1张图片的总人数
        num_yolo = yolo.returnNum("restaurant", 3)
        num = p * num_retinanet + q * num_mcnn + r * num_yolo
        self.res_result_3.setText(str(format(num, ".2f")))
        num_mcnn = mcnn.returnNum("restaurant", 4)  # 用mcnn获取第1张图片的总人数
        num_retinanet = retinanet.returnNum("restaurant", 4)  # 用RetinaNet获取第1张图片的总人数
        num_yolo = yolo.returnNum("restaurant", 4)
        num = p * num_retinanet + q * num_mcnn + r * num_yolo
        self.res_result_4.setText(str(format(num, ".2f")))
        num_mcnn = mcnn.returnNum("restaurant", 5)  # 用mcnn获取第1张图片的总人数
        num_retinanet = retinanet.returnNum("restaurant", 5)  # 用RetinaNet获取第1张图片的总人数
        num_yolo = yolo.returnNum("restaurant", 5)
        num = p * num_retinanet + q * num_mcnn + r * num_yolo
        self.res_result_5.setText(str(format(num, ".2f")))
        num_mcnn = mcnn.returnNum("restaurant", 6)  # 用mcnn获取第1张图片的总人数
        num_retinanet = retinanet.returnNum("restaurant", 6)  # 用RetinaNet获取第1张图片的总人数
        num_yolo = yolo.returnNum("restaurant", 6)
        num = p * num_retinanet + q * num_mcnn + r * num_yolo
        self.res_result_6.setText(str(format(num, ".2f")))

        num_mcnn = mcnn.returnNum("outdoor", 1)  # 用mcnn获取第1张图片的总人数
        num_retinanet = retinanet.returnNum("outdoor", 1)  # 用RetinaNet获取第1张图片的总人数
        num_yolo = yolo.returnNum("outdoor", 1)
        num = p * num_retinanet + q * num_mcnn + r * num_yolo
        self.out_result_1.setText(str(format(num, ".2f")))
        num_mcnn = mcnn.returnNum("outdoor", 2)  # 用mcnn获取第1张图片的总人数
        num_retinanet = retinanet.returnNum("outdoor", 2)  # 用RetinaNet获取第1张图片的总人数
        num_yolo = yolo.returnNum("outdoor", 2)
        num = p * num_retinanet + q * num_mcnn + r * num_yolo
        self.out_result_2.setText(str(format(num, ".2f")))
        num_mcnn = mcnn.returnNum("outdoor", 3)  # 用mcnn获取第1张图片的总人数
        num_retinanet = retinanet.returnNum("outdoor", 3)  # 用RetinaNet获取第1张图片的总人数
        num_yolo = yolo.returnNum("outdoor", 3)
        num = p * num_retinanet + q * num_mcnn + r * num_yolo
        self.out_result_3.setText(str(format(num, ".2f")))

        num_mcnn = mcnn.returnNum("classroom", 1)  # 用mcnn获取第1张图片的总人数
        num_retinanet = retinanet.returnNum("classroom", 1)  # 用RetinaNet获取第1张图片的总人数
        num_yolo = yolo.returnNum("classroom", 1)
        num = p * num_retinanet + q * num_mcnn + r * num_yolo
        self.class_result_1.setText(str(format(num, ".2f")))
        num_mcnn = mcnn.returnNum("classroom", 2)  # 用mcnn获取第1张图片的总人数
        num_retinanet = retinanet.returnNum("classroom", 2)  # 用RetinaNet获取第1张图片的总人数
        num_yolo = yolo.returnNum("classroom", 2)
        num = p * num_retinanet + q * num_mcnn + r * num_yolo
        self.class_result_2.setText(str(format(num, ".2f")))
        num_mcnn = mcnn.returnNum("classroom", 3)  # 用mcnn获取第1张图片的总人数
        num_retinanet = retinanet.returnNum("classroom", 3)  # 用RetinaNet获取第1张图片的总人数
        num_yolo = yolo.returnNum("classroom", 3)
        num = p * num_retinanet + q * num_mcnn + r * num_yolo
        self.class_result_3.setText(str(format(num, ".2f")))
        num_mcnn = mcnn.returnNum("classroom", 4)  # 用mcnn获取第1张图片的总人数
        num_retinanet = retinanet.returnNum("classroom", 4)  # 用RetinaNet获取第1张图片的总人数
        num_yolo = yolo.returnNum("classroom", 4)
        num = p * num_retinanet + q * num_mcnn + r * num_yolo
        self.class_result_4.setText(str(format(num, ".2f")))
        num_mcnn = mcnn.returnNum("classroom", 5)  # 用mcnn获取第1张图片的总人数
        num_retinanet = retinanet.returnNum("classroom", 5)  # 用RetinaNet获取第1张图片的总人数
        num_yolo = yolo.returnNum("classroom", 5)
        num = p * num_retinanet + q * num_mcnn + r * num_yolo
        self.class_result_5.setText(str(format(num, ".2f")))
        num_mcnn = mcnn.returnNum("classroom", 6)  # 用mcnn获取第1张图片的总人数
        num_retinanet = retinanet.returnNum("classroom", 6)  # 用RetinaNet获取第1张图片的总人数
        num_yolo = yolo.returnNum("classroom", 6)
        num = p * num_retinanet + q * num_mcnn + r * num_yolo
        self.class_result_6.setText(str(format(num, ".2f")))

    #清空人数
    def clear_num(self):
        self.res_result_1.setText("0")
        self.res_result_2.setText("0")
        self.res_result_3.setText("0")
        self.res_result_4.setText("0")
        self.res_result_5.setText("0")
        self.res_result_6.setText("0")
        self.out_result_1.setText("0")
        self.out_result_2.setText("0")
        self.out_result_3.setText("0")
        self.class_result_1.setText("0")
        self.class_result_2.setText("0")
        self.class_result_3.setText("0")
        self.class_result_4.setText("0")
        self.class_result_5.setText("0")
        self.class_result_6.setText("0")

    #清空图像
    def clear_image(self):
        self.res_image_1.setText("请单击“加载图像”")
        self.res_image_2.setText("请单击“加载图像”")
        self.res_image_3.setText("请单击“加载图像”")
        self.res_image_4.setText("请单击“加载图像”")
        self.res_image_5.setText("请单击“加载图像”")
        self.res_image_6.setText("请单击“加载图像”")
        self.out_image_1.setText("请单击“加载图像”")
        self.out_image_2.setText("请单击“加载图像”")
        self.out_image_3.setText("请单击“加载图像”")
        self.class_image_1.setText("请单击“加载图像”")
        self.class_image_2.setText("请单击“加载图像”")
        self.class_image_3.setText("请单击“加载图像”")
        self.class_image_4.setText("请单击“加载图像”")
        self.class_image_5.setText("请单击“加载图像”")
        self.class_image_6.setText("请单击“加载图像”")

if __name__ == "__main__":
    app = QtGui.QApplication(argv)
    window = MyApp()
    window.show()
    exit(app.exec_())