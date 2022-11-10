from pynput import keyboard
from pynput.mouse import Button, Controller as c_mouse
import time
import msvcrt
import keyboard
import sys
import  pyautogui
from PIL import ImageGrab
import win32api
import win32con
import win32gui
import win32api, win32gui, win32print
from win32.lib import win32con




mouse= c_mouse()
isOver = True
a = time.time() +60
b= 0
flag =True
def ListenKey(x):
    global flag
    a = keyboard.KeyboardEvent(event_type='down', scan_code=1, name='esc')

    # # get key code and name
    # print("current key code:  {}".format(x.scan_code))
    # print("current key name:  {}".format(x.name))

    if x.event_type == a.event_type and x.scan_code == a.scan_code:
        print("You pressed {}.".format(a.name))
        flag = False
        sys.exit(0 )
keyboard.hook(ListenKey)
hDC = win32gui.GetDC(0)
sX = win32print.GetDeviceCaps(hDC, win32con.DESKTOPHORZRES)   #获得屏幕分辨率X轴
sY = win32print.GetDeviceCaps(hDC, win32con.DESKTOPVERTRES)   #获得屏幕分辨率Y轴

print(sX ,sY)
while flag:
    #a = time.time()
#captureImage = ImageGrab.grab(bbox=(600, 300, 1000, 600))
#captureImage.show()
    mouse.position = (800, 450)
    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, 100, 0)



    print(win32api.GetCursorPos())
    time.sleep(0.02)
    #print(time.time()-a)

#while a- time.time()>0:
 #   b = b+1
    #if (a - time.time() < 5):
        #mouse.position = (592, 254)

##按下右键

#mouse.press(Button.right)





##释放右键
#mouse.release(Button.right)while isOver:
#114 199 58 200
#614 214


