import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

# UI 관련 import
from enum import auto
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import uic
from PyQt5 import QtCore
from PyQt5.QtCore import QObject
import cv2
#import time
import Script1
import gspeech
import datetime
from pytz import timezone
import serial1 as se
import time
import urllib.request
# 이건 임시 test용 Firebase에서 정보를 가져오는 코드
cred = credentials.Certificate('iotapp-5d4d0-firebase-adminsdk-a2zrl-cb61a099d6.json')
firebase_admin.initialize_app(cred,{
    'databaseURL' : 'https://console.firebase.google.com/project/iotapp-5d4d0/firestore/data/~2F',
    'storageBucket' : "gs://iotapp-5d4d0.appspot.com " 
    #'databaseURL' : '데이터 베이스 url'
})
db = firestore.client()

# UI를 불러오는 코드, UI파일 이름은 untitled.ui 
formClass = uic.loadUiType("untitled.ui")[0]

class Ui(QMainWindow, formClass): ##JK 12050400

# 기초 init 설정 
    def __init__(self):
        super().__init__()
        # Use Thread
        # self.test = Test()
        # self.test_thread = QtCore.QThread()
        # self.test.moveToThread(self.test_thread)
        # self.test_thread.start()
        # self.thread = QtCore.QThread()
        # self.thread.start()
        # self.test = Test()
        # self.test.moveToThread(self.thread)
        #self.test = Test()
        self.image_2 = cv2.imread("./smilingface.jpg", cv2.IMREAD_COLOR)
        self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
        self.setupUi(self)
        self.StartSetting()

# 기초 시작코드, 버튼 관련 함수 연결 등, 버튼 관련 함수는 모두 여기에 써야함 안 그럴 시 thread 무한 생성
    def StartSetting(self):
        self.setFixedSize(QSize(800, 600))
        self.stackedWidget.setCurrentIndex(0)
        self.pushButton.clicked.connect(self.Login)
        self.pushButton_5.clicked.connect(self.PreparePage)
        self.pushButton_7.clicked.connect(self.SmilingFace)
        self.pushButton_8.clicked.connect(self.SettingPage)
        self.pushButton_9.clicked.connect(self.LoginPage)
        self.pushButton_10.clicked.connect(self.PreparePage)
        

# Login부분은 UID만 입력
    def Login(self):
        # uid 전역변수 설정
        global uid
        input_id = db.collection(u'Res').document(self.lineEdit.text())
        is_id_exist = input_id.get()
        uid = self.lineEdit.text()

        # print(self.lineEdit.text())
        if is_id_exist.exists:
            # if db.collection(u'Res')
            SuccessMsg = '로그인 성공'
            QMessageBox.about(self,"성공",SuccessMsg)

            # # Use Thread
            self.test = Test()
            # self.test_thread = QtCore.QThread()
            # if self.test_thread.isRunning(): 
            #     self.test_thread.terminate()
            # self.speech = Speech()
            # self.test.moveToThread(self.test_thread)
            # self.test_thread.start()
            # self.speech.moveToThread(self.thread)
            self.PreparePage()
        else:
            errorMsg = '로그인 실패'
            QMessageBox.about(self,"실패",errorMsg)
        self.pushButton_6.clicked.connect(self.test.Eaten)
# 얘는 smiling face를 화면에 띄우도록하는 코드
    def SmilingFace(self):
        self.stackedWidget.setCurrentIndex(2)
        # image_2 = cv2.imread("./smilingface.jpg", cv2.IMREAD_COLOR)
        h,w = self.image_2.shape[:2]
        image_2 = cv2.cvtColor(self.image_2,cv2.COLOR_BGR2RGB)
        qt_img = QImage(image_2, w, h, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qt_img) 
        self.label_7.setPixmap(pix.scaled(self.label_7.size(), Qt.IgnoreAspectRatio))
        self.test.start_test()
        # self.do_work()

    #Login Page로 돌아감
    def LoginPage(self) :
        self.stackedWidget.setCurrentIndex(0)
        self.test.stop_test()
        # self.test_thread.terminate()
        # self.test_thread.wait()       
        # self.test_thread.start()
        # self.speech.stop_test2()

    #Setting Page로 감
    def PreparePage(self):
        self.stackedWidget.setCurrentIndex(4)

    def SettingPage(self):
        self.stackedWidget.setCurrentIndex(5)

    # def do_work(self):
    #     Script1.s_main()
    #     self.finished.emit()

    # def test(self):
    #     doc_ref = db.collection(u'Res').document(u'uidtest')
    #     docs = doc_ref.get()
    #     s1 = str(docs.to_dict())
    #     s2 = s1.split(":")
    #     s3 = s2[1].split(", ")
    #     s4 = s3[0].split("(")
    #     print(s3[2])
    #     if s3[2] == str(26):
    #         print("yes")
    #     else:
    #         print("no")
        # Script1.func1()

# gspeech and Get Appointment Time           
class Test(QThread):
    def __init__(self):
        super().__init__()
        self.cnt = 0
        self.stop_flag = False
        self.update_reserv = 0
        self.tmp_time = -1
        self.eaten = False
        self.pagemoved = True
        self.Test = True
        self.url = None
        # self.speech = Speech()
        # self.speech.moveToThread(self.thread)

    def start_test(self):
        print(uid)
        self.cnt = 0
        self.stop_flag = False
        self.update_reserv = 0
        now_in_charge = -1
        KST = timezone('Asia/Seoul')
        while True:
            self.cnt += 1
            print('test1 = ', self.cnt)
            # if self.cnt == 2 or self.cnt == 10:
            #     gspeech.main()
                # self.speech.start_test2()
            
            print(db.collection(u'Users').document(uid).get()._data.get('url'))
            if self.url != db.collection(u'Users').document(uid).get()._data.get('url'):
                self.url = db.collection(u'Users').document(uid).get()._data.get('url')
                if self.url == None:
                    image_2 = cv2.imread("./smilingface.jpg", cv2.IMREAD_COLOR)
                    h,w = image_2.shape[:2]
                    image_2 = cv2.cvtColor(image_2,cv2.COLOR_BGR2RGB)
                    qt_img = QImage(image_2, w, h, QImage.Format_RGB888)
                    pix = QPixmap.fromImage(qt_img) 
                    currentUi.label_7.setPixmap(pix.scaled(currentUi.label_7.size(), Qt.IgnoreAspectRatio))
                    currentUi.test.start_test()
                else:
                    image = urllib.request.urlopen(self.url).read()
                pixmap = QPixmap()
                pixmap.loadFromData(image)
                currentUi.label_7.setPixmap(pixmap.scaled(currentUi.label_7.size(), Qt.IgnoreAspectRatio))
                currentUi.stackedWidget.setCurrentIndex(2)
            tm = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
            tm_timestamp = datetime.datetime.now().timestamp()
            doc1_ref = db.collection(u'Todo')
            doc_ref = doc1_ref.document(uid)
            docs = doc_ref.get()
            print("docs: ",docs, "type: ",type(docs))
            # s1 = str(docs.to_dict())
            # s1_dict = docs._data
            
            if docs._data.get('morning') != None:
                morning = docs._data.get('morning').timestamp()
                morning_timestamp = docs._data.get('morning').astimezone(KST).strftime("%Y-%m-%d-%H-%M")
            else:
                morning = 9999999999.9
                morning_timestamp = '9999-99-99-99-99'
            if docs._data.get('lunch') != None:
                lunch = docs._data.get('lunch').timestamp() 
                lunch_timestamp = docs._data.get('lunch').astimezone(KST).strftime("%Y-%m-%d-%H-%M")
            else:
                lunch = 9999999999.9
                lunch_timestamp = '9999-99-99-99-99'
            if docs._data.get('dinner') != None:
                dinner = docs._data.get('dinner').timestamp() # time stamp 형 float
                dinner_timestamp = docs._data.get('dinner').astimezone(KST).strftime("%Y-%m-%d-%H-%M") # 2222-22-22-22-22-22 str형
            else:
                dinner = 9999999999.9
                dinner_timestamp = '9999-99-99-99-99'

            mld = [morning, lunch, dinner]
            for item in mld :
                if item != 9999999999.9:
                    now_in_charge = item
                    if mld.index(item) == 0:
                        now_in_charge_timestamp = morning_timestamp
                        now_in_charge_name = 'morning'
                    elif mld.index(item) == 1:
                        now_in_charge_timestamp = lunch_timestamp
                        now_in_charge_name = 'lunch'
                    else:
                        now_in_charge_timestamp = dinner_timestamp
                        now_in_charge_name = 'dinner'
                    break
            if now_in_charge == -1:
                now_in_charge = 9999999999.9
                now_in_charge_timestamp = '9999-99-99-99-99'
                #self.update_reserv = 1
            
            print(now_in_charge, type(now_in_charge))
            print(now_in_charge_timestamp, type(now_in_charge_timestamp))
            print(tm, type(tm))
            
            #print(resv_time, type(resv_time))
            # 복약여부 확인 전
            if self.update_reserv == 0:
                
                # 예약시간 == 현재시간 시 음성인식 실행
                ######################################################
                if tm == now_in_charge_timestamp:
                    #####################################################
                    # self.update_reserv = 1
                    if self.pagemoved == True:
                        if now_in_charge == morning:
                            se.motor_move('a')
                        elif now_in_charge == lunch:
                            se.motor_move('b')
                        elif now_in_charge == dinner:
                            se.motor_move('c')
                    print('복약여부확인 음성 받기 시작')
                    # print(gspeech.main())
                    # 음성인식
                    medicine = gspeech.main()
                    # '먹었어' 대답 시 True 값 전달
                    if medicine == 1:
                        #doc1_ref.document(u'update').update({u'eat': True})
                        ###먹었다
                        reservation_data = db.collection(u'Res').document(uid).get().to_dict()
                        for key, value in reservation_data.items():
                            if value == {}:
                                continue
                            else:
                                for key2, value2 in value.items():
                                    if key2 == now_in_charge_timestamp[0:10]:
                                        datafile_big = db.collection(u'Res').document(uid).get()._data
                                        datafile = datafile_big.get(key)
                                        datafile.get(key2)['is'+str(now_in_charge_name)] = 'done'
                                        db.collection(u'Res').document(uid).update({key : datafile})
                                        time.sleep(10)
                                        db.collection(u'Todo').document(uid).update({now_in_charge_name : None})
                                        self.pagemoved = True
                                        now_in_charge = 9999999999.9
                                        now_in_charge_timestamp = '9999-99-99-99-99'
                                        self.update_reserv = 0
                                                                                                                                        
                            #             tmp_ref = db.collection(u'Res').document(uid)._data.get().to_dict()
                        ###
                        time.sleep(10)
                        continue
                    # 대답이 없거나, '먹었어'를 대답 안할 시 False 값 전달
                    elif medicine == 2:
                        time.sleep(20)
                        ###안먹었다
                        self.pagemoved = False
                        self.Movepage()
                        #self.update_reserv = 1
                        continue
                    # print(gspeech.gsp.cnt)
                    # time.sleep(5)
                    
            if self.eaten == True:
                self.eaten = False
                #self.update_reserv = 1
                reservation_data = db.collection(u'Res').document(uid).get().to_dict()
                for key, value in reservation_data.items():
                    if value == {}:
                        continue
                    else:
                        for key2, value2 in value.items():
                            if key2 == now_in_charge_timestamp[0:10]:
                                datafile_big = db.collection(u'Res').document(uid).get()._data
                                datafile = datafile_big.get(key)
                                datafile.get(key2)['is'+str(now_in_charge_name)] = 'done'
                                db.collection(u'Res').document(uid).update({key : datafile})
                                time.sleep(10)
                                db.collection(u'Todo').document(uid).update({now_in_charge_name : None})
                                self.pagemoved = True
                                now_in_charge = 9999999999.9
                                now_in_charge_timestamp = '9999-99-99-99-99'
                                self.update_reserv = 0
                                
            # # 복약여부 확인 후 예약시간 보다 3 분이 지나면 다시 예약시간과 현재시간 비교
            # elif self.update_reserv == 1 and take == 0 :
            #     print('stop')
            #     if (tm.tm_min - self.tmp_time) == 1 or ( tm.tm_min - self.tmp_time ) == -59:
            #         self.update_reserv = 0
            if (self.eaten == False) and now_in_charge+180 < tm_timestamp and tm_timestamp < now_in_charge+185:
                #self.update_reserv = 0
                self.Test = True

            loop = QtCore.QEventLoop()
            QtCore.QTimer.singleShot(1000, loop.quit) #1000 ms
            loop.exec_()

            if self.stop_flag:
                self.stop_flag = False
                break
    def Eaten(self):
        self.eaten = True
        currentUi.stackedWidget.setCurrentIndex(2)

    def stop_test(self):
        print('finish')
        self.stop_flag = True
        # return

    def Movepage(self):
        if self.pagemoved == False:
            currentUi.stackedWidget.setCurrentIndex(3)
            self.pagemoved = True
# class Speech(QObject):
#     def __init__(self):
#         super().__init__()
#         self.cnt2 = 0
#         self.stop_flag2 = False

#     def start_test2(self):
#         while True:
#             self.cnt2 += 1
#             print('test2 = ', self.cnt2)

#             loop = QtCore.QEventLoop()
#             QtCore.QTimer.singleShot(1000, loop.quit) #1000 ms
#             loop.exec_()
#             if self.cnt2 == 5:
#                 break
#             if self.stop_flag2:
#                 self.stop_flag2 = False
#                 break

#     def stop_test2(self):
#         print('finish')
#         self.stop_flag2 = True


# 실제 code의 실행은 이부분에서
if __name__ == '__main__':
    app = QApplication(sys.argv)
    currentUi = Ui()
    currentUi.showMaximized()
    sys.exit(app.exec_())





# --------------이 밑부분은 무시, 메모장용---------------

# 리소스 변경 시마다 터미널창에 입력
# pyrcc5 resource.qrc -o resource_rc.py

# 실행파일로 만들기
# pyinstaller -w -F logincopy.py
