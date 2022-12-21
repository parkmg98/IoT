import os
import cv2
import time
import torch
import argparse
import numpy as np
import Jetson.GPIO as GPIO

from Detection.Utils import ResizePadding
from CameraLoader import CamLoader, CamLoader_Q
from DetectorLoader import TinyYOLOv3_onecls

from PoseEstimateLoader import SPPE_FastPose
from fn import draw_single

from Track.Tracker import Detection, Tracker
from ActionsEstLoader import TSSTG

'''
( pip install firebase-admin )
'''
import datetime
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

cred = credentials.Certificate('/home/hi/Human-Falling-Detect-Tracks/iotapp-5d4d0-firebase-adminsdk-a2zrl-cb61a099d6.json')
firebase_admin.initialize_app(cred,{
    'databaseURL' : 'https://console.firebase.google.com/project/iotapp-5d4d0/firestore/data/~2F' 
    #'databaseURL' : '데이터 베이스 url'
})
db = firestore.client()


# 추후에 UID 수정
UID = '3XZWaY6znufBVQ4aBZQr2eRYSho2'

doc_ref = db.collection(u'Fall').document(UID)
doc_ref.set({
    u'is_fall': False
})

fall_list = [0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0]
is_fall_change = 1

# servo motor setup
SERVO_PIN_1 = 33
SERVO_PIN_2 = 32

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(SERVO_PIN_1, GPIO.OUT)
GPIO.setup(SERVO_PIN_2, GPIO.OUT)

pwm_1 = GPIO.PWM(SERVO_PIN_1, 50)
pwm_2 = GPIO.PWM(SERVO_PIN_2, 50)

pwm_1.start((1./18.)*120 + 2)
pwm_2.start((1./18.)*120 + 2)

# motor angle setup
global motor_angle1
#motor_angle1 = 120

#source = '../Data/test_video/test7.mp4'
#source = '../Data/falldata/Home/Videos/video (2).avi'  # hard detect
source = '../Data/falldata/Home/Videos/video (1).avi'
#source = 2
bbox = [0,0,0,0]

def preproc(image):
    """preprocess function for CameraLoader.
    """
    image = resize_fn(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def kpt2bbox(kpt, ex=20):
    """Get bbox that hold on all of the keypoints (x,y)
    kpt: array of shape `(N, 2)`,
    ex: (int) expand bounding box,
    """
    return np.array((kpt[:, 0].min() - ex, kpt[:, 1].min() - ex,
                     kpt[:, 0].max() + ex, kpt[:, 1].max() + ex))


if __name__ == '__main__':
    par = argparse.ArgumentParser(description='Human Fall Detection Demo.')
    par.add_argument('-C', '--camera', default=source,  # required=True,  # default=2,
                        help='Source of camera or video file path.')
    par.add_argument('--detection_input_size', type=int, default=384,
                        help='Size of input in detection model in square must be divisible by 32 (int).')
    par.add_argument('--pose_input_size', type=str, default='224x160',
                        help='Size of input in pose model must be divisible by 32 (h, w)')
    par.add_argument('--pose_backbone', type=str, default='resnet50',
                        help='Backbone model for SPPE FastPose model.')
    par.add_argument('--show_detected', default=False, action='store_true',
                        help='Show all bounding box from detection.')
    par.add_argument('--show_skeleton', default=True, action='store_true',
                        help='Show skeleton pose.')
    par.add_argument('--save_out', type=str, default='',
                        help='Save display to video file.')
    par.add_argument('--device', type=str, default='cuda',
                        help='Device to run model on cpu or cuda.')
    args = par.parse_args()

    device = args.device

    # DETECTION MODEL.
    inp_dets = args.detection_input_size
    detect_model = TinyYOLOv3_onecls(inp_dets, device=device)

    # POSE MODEL.
    inp_pose = args.pose_input_size.split('x')
    inp_pose = (int(inp_pose[0]), int(inp_pose[1]))
    pose_model = SPPE_FastPose(args.pose_backbone, inp_pose[0], inp_pose[1], device=device)

    # Tracker.
    max_age = 30
    tracker = Tracker(max_age=max_age, n_init=3)

    # Actions Estimate.
    action_model = TSSTG()

    resize_fn = ResizePadding(inp_dets, inp_dets)

    cam_source = args.camera
    if type(cam_source) is str and os.path.isfile(cam_source):
        # Use loader thread with Q for video file.
        cam = CamLoader_Q(cam_source, queue_size=1000, preprocess=preproc).start()
    else:
        # Use normal thread loader for webcam.
        cam = CamLoader(int(cam_source) if cam_source.isdigit() else cam_source,
                        preprocess=preproc).start()

    #frame_size = cam.frame_size
    #scf = torch.min(inp_size / torch.FloatTensor([frame_size]), 1)[0]

    outvid = False
    if args.save_out != '':
        outvid = True
        codec = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter(args.save_out, codec, 30, (inp_dets * 2, inp_dets * 2))

    fps_time = 0
    f = 0
    fps_number = 0
    fall_count = 0
    motor_angle1 = 120
    motor_number = 0

    while cam.grabbed():
        print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        f += 1
        fps_number += 1
        frame = cam.getitem()
        image = frame.copy()
        fall_count += 1
        motor_number += 1
        # print('test')

        # Detect humans bbox in the frame with detector model.
        detected = detect_model.detect(frame, need_resize=False, expand_bb=10)

        # Predict each tracks bbox of current frame from previous frames information with Kalman filter.
        tracker.predict()
        # Merge two source of predicted bbox together.
        for track in tracker.tracks:
            det = torch.tensor([track.to_tlbr().tolist() + [0.5, 1.0, 0.0]], dtype=torch.float32)
            detected = torch.cat([detected, det], dim=0) if detected is not None else det

        detections = []  # List of Detections object for tracking.
        if detected is not None:
            #detected = non_max_suppression(detected[None, :], 0.45, 0.2)[0]
            # Predict skeleton pose of each bboxs.
            poses = pose_model.predict(frame, detected[:, 0:4], detected[:, 4])

            # Create Detections object.
            detections = [Detection(kpt2bbox(ps['keypoints'].numpy()),
                                    np.concatenate((ps['keypoints'].numpy(),
                                                    ps['kp_score'].numpy()), axis=1),
                                    ps['kp_score'].mean().numpy()) for ps in poses]

            # VISUALIZE.
            if args.show_detected:
                for bb in detected[:, 0:5]:
                    frame = cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 0, 255), 1)

        # Update tracks by matching each track information of current and previous frame or
        # create a new track if no matched.
        tracker.update(detections)
        print(fall_count, fps_number, motor_number)

        # Rotate the motor one turn if not found human for a certainn amount of time
        if fall_count == 200:
            # move motor 90
            DC_1 = (1./18.)*90 + 2     
            pwm_1.ChangeDutyCycle(DC_1)
            time.sleep(1.5)
            # move motor 60
            DC_1 = (1./18.)*60 + 2     
            pwm_1.ChangeDutyCycle(DC_1)
            motor_angle=60
            time.sleep(3)      

        if fall_count == 260:
            # move motor 150
            DC_1 = (1./18.)*150 + 2     
            pwm_1.ChangeDutyCycle(DC_1)
            time.sleep(1.5)
            # move motor 180
            DC_1 = (1./18.)*180 + 2      
            pwm_1.ChangeDutyCycle(DC_1)
            motor_angle=180
            time.sleep(3)          

        if fall_count == 320:
            # move motor center
            DC_1 = (1./18.)*120 + 2      
            pwm_1.ChangeDutyCycle(DC_1)
            motor_angle=120
            time.sleep(1)              
            
            # reset fall_count
            fall_count = 0
       
        # 0: nothing 1: walkings 2: sittings 3: lying 4: fall
        #fall_list = [0, 0, 0, 0, 0, 0, 0, 0,0,0,0,0,0,0,0,0]
        fall_list.pop(0)
        fall_list.insert(16,0)

        # send fall message to server
        if fall_list.count(4) >= 11:
            #if fall_list.count(1) < 5 and fall_list.count(2) < 5:
            if fall_list[2] == 4 and (fall_list[0] == 1 or fall_list[1] == 1):
                print("fall!!\n\n")
                doc_ref = db.collection(u'Users').document(UID)
                doc_ref.update({
                    u'is_fall': True
                })
                is_fall_change = 0

        # Predict Actions of each track.
        for i, track in enumerate(tracker.tracks):
            #global motor_angle1
            motor_angle=0
            #motor_angle120=120
            fall_count = 0
            if not track.is_confirmed():
                print('nothing')
                continue

            track_id = track.track_id
            bbox = track.to_tlbr().astype(int)
            center = track.get_center().astype(int)

            action = 'pending..'
            clr = (0, 255, 0)
            # Use 30 frames time-steps to prediction.
            if len(track.keypoints_list) == 30:
                pts = np.array(track.keypoints_list, dtype=np.float32)
                out = action_model.predict(pts, frame.shape[:2])
                action_name = action_model.class_names[out[0].argmax()]
                action = '{}: {:.2f}%'.format(action_name, out[0].max() * 100)
                if action_name == 'Fall Down':
                    clr = (255, 0, 0)
                    fall_list.pop(15)
                    fall_list.insert(16,4)
                    #if is_fall_change == 1:                        
                        #doc_ref = db.collection(u'Users').document(UID)
                        #doc_ref.update({
                        #    u'is_fall': True
                        #})
                    is_fall_change = 0
                elif action_name == 'Lying Down':
                    fall_list.pop(15)
                    fall_list.insert(16,3)
                    clr = (255, 200, 0)
                elif action_name == 'Walking' or action_name == 'Standing' or action_name == 'Stand up':
                    fall_list.pop(15)
                    fall_list.insert(16,1)
                    is_fall_change = 1
                elif action_name == 'Sit down'or action_name == 'Sitting':
                    fall_list.pop(15)
                    fall_list.insert(16,2)
                    is_fall_change = 1
                else:
                    is_fall_change = 1

            print('bbx[0]: ',bbox[0], 'bbx[1]: ',bbox[1], 'bbx[2]: ',bbox[2], 'bbx[3]: ',bbox[3], '(bbox[0]+bbox[2])/2: ', (bbox[0]+bbox[2])/2)

            # angle form center
            angle = -(((bbox[0]+bbox[2])/2.-70.)-120.)/9.
            print('angle:', round(angle,2))
            #fall_count = fall_count+angle
            #print('fall_count: ', fall_count)
            #motor_angle = motor_angle1 + angle
            #tests = motor_angle1 + 10
            #print('tests:', tests)
            #print('angle:', round(angle,2), 'motor_angle1:', round(motor_angle1,2), 'motor_angle:', round(motor_angle,2))

            # catching off-screen bbox (center == 190)
            if bbox[2]>bbox[0] and (bbox[0]<40 or bbox[2]>340) and ((bbox[0]+bbox[2])/2>240) or ((bbox[0]+bbox[2])/2<140):
                #motor_angle = motor_angle1 + angle
                #print('angle:', round(angle,2), 'motor_angle1:', round(motor_angle1,2), 'motor_angle:', round(motor_angle,2))

                # run every frame (hardware dependent)
                if motor_number >= 10:
                    motor_angle = motor_angle1 + angle
                    print('angle:', round(angle,2), 'motor_angle1:', round(motor_angle1,2), 'motor_angle:', round(motor_angle,2))
                    # limit angle
                    if (motor_angle<180) and (motor_angle>60):
                        DC_1 = (1./18.)*motor_angle + 2
                        # move motor 
                        pwm_1.ChangeDutyCycle(DC_1)
                        print('(motor_angle<180) and (motor_angle>60)')
                    elif motor_angle>=180:
                        motor_angle = 180
                    #    DC_1 = (1./18.)*140 + 2
                    #    pwm_1.ChangeDutyCycle(DC_1)
                        print('motor_angle>=180')
                    elif motor_angle<=60:
                        motor_angle = 60
                    #     DC_1 = (1./18.)*100 + 2
                    #     pwm_1.ChangeDutyCycle(DC_1)
                        print('motor_angle<=60')
                    motor_number = 0
            #frame = cv2.rectangle(frame, (0,0), (190, 190), (0, 255, 0), 1)


            # VISUALIZE.
            if track.time_since_update == 0:
                if args.show_skeleton:
                    frame = draw_single(frame, track.keypoints_list[-1])
                frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)
                frame = cv2.putText(frame, str(track_id), (center[0], center[1]), cv2.FONT_HERSHEY_COMPLEX,
                                    0.4, (255, 0, 0), 2)
                frame = cv2.putText(frame, action, (bbox[0] + 5, bbox[1] + 15), cv2.FONT_HERSHEY_COMPLEX,
                                    0.4, clr, 1)

            if motor_angle != 0:
                motor_angle1=motor_angle
                print('fixed motor_angle1: ',round(motor_angle1,2))
        # frame = cv2.rectangle(frame, (200,0), (380, 200), (0, 255, 0), 1)
        print('fall_list: ',fall_list)
        # Show Frame.
        frame = cv2.resize(frame, (0, 0), fx=2., fy=2.)
        frame = cv2.putText(frame, '%d, FPS: %f' % (f, 1.0 / (time.time() - fps_time)),
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        frame = frame[:, :, ::-1]
        fps_time = time.time()

        if outvid:
            writer.write(frame)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clear resource.
    cam.stop()
    pwm_1.stop()
    pwm_2.stop()
    GPIO.cleanup()
    if outvid:
        writer.release()
    cv2.destroyAllWindows()
