from datetime import datetime
import time

import keyboard

print()
dts = datetime.now()
flag = False


def abc(x):
    global flag
    a = keyboard.KeyboardEvent(event_type='down', scan_code=57, name='space')

    # # get key code and name
    # print("current key code:  {}".format(x.scan_code))
    # print("current key name:  {}".format(x.name))

    if x.event_type == a.event_type and x.scan_code == a.scan_code:
        print("You pressed {}.".format(a.name))
        flag = True


keyboard.hook(abc)
num_iter = 20

for i in range(num_iter):
    print(i)
    time.sleep(0.2)
    if flag:
        print("Paused, please input something:")
        ipt = input()
        print("You input {}".format(ipt))
        flag = False

dte = datetime.now()
tm = round((dte - dts).seconds + (dte - dts).microseconds / (10 ** 6), 3)
print("total running time:  {} s".format(tm), '\n')







