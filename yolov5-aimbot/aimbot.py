#关闭鼠标加速
#1600*900
import torch
from PIL import ImageGrab
import keyboard
from pynput.mouse import Button, Controller as c_mouse
import win32api, win32gui, win32print
from win32.lib import win32con
import time



def main():
    mouse = c_mouse()
    flag = True
    






    def ListenKey(x):
        global flag
        a = keyboard.KeyboardEvent(event_type='down', scan_code=1, name='esc')

        # # get key code and name
        # print("current key code:  {}".format(x.scan_code))
        # print("current key name:  {}".format(x.name))

        if x.event_type == a.event_type and x.scan_code == a.scan_code:
            print('Aimbot finish')
            flag = False
    keyboard.hook(ListenKey)


    #获得屏幕分辨率
    hDC = win32gui.GetDC(0)
    sX = win32print.GetDeviceCaps(hDC, win32con.DESKTOPHORZRES)
    sY = win32print.GetDeviceCaps(hDC, win32con.DESKTOPVERTRES)


    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    print('aimbot start')
    while flag:

        captureImage = ImageGrab.grab()

        results = model(captureImage)
        x1 = results.pandas().xyxy[0]['xmin']
        y1 = results.pandas().xyxy[0]['ymin']
        x2 = results.pandas().xyxy[0]['xmax']
        y2 = results.pandas().xyxy[0]['ymax']
        labels = results.pandas().xyxy[0]['name']
        for label in labels:
            i = 0
            if str(label) =='person':
                Xmin,Ymin,Xmax,Ymax =x1[i],y1[i],x2[i],y2[i]

                x = int((Xmin + Xmax) / 2)
                y = int((Ymin + Ymax) / 2)

                X = int(x - sX /2)
                Y = int(y - sY/2)



                print(label+' is in '+str((x,y))+',mouse center is in '+str((sX / 2, sY/2))+' it should move ' + str((X, Y)) )

                print('')

                mouse.position = (sX / 2, sY / 2)

                win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, X, Y)

                time.sleep(0.02)


            i = i+1






if __name__ == "__main__":
    main()
