#!/usr/bin/env python3
"""
Simulates a natural circular 'Circle to Select' touch gesture on Snapdragon 8 Elite.
"""

import math
import subprocess
import time

def send_circle_gesture(dev="/dev/input/event6", cx=540, cy=991, r=160, steps=45, duration=0.8):
    # EV codes
    EV_SYN = 0
    SYN_REPORT = 0
    EV_KEY = 1
    BTN_TOUCH = 330
    EV_ABS = 3
    ABS_X = 0
    ABS_Y = 1
    ABS_MT_TOUCH_MAJOR = 48
    ABS_MT_POSITION_X = 53
    ABS_MT_POSITION_Y = 54
    ABS_MT_TRACKING_ID = 57

    cmds = []
    
    # Finger Down
    theta0 = 0.0
    x0 = int(cx + r * math.cos(theta0))
    y0 = int(cy + r * math.sin(theta0))
    
    cmds.append(f"sendevent {dev} {EV_ABS} {ABS_MT_TRACKING_ID} 101")
    cmds.append(f"sendevent {dev} {EV_KEY} {BTN_TOUCH} 1")
    cmds.append(f"sendevent {dev} {EV_ABS} {ABS_MT_POSITION_X} {x0}")
    cmds.append(f"sendevent {dev} {EV_ABS} {ABS_MT_POSITION_Y} {y0}")
    cmds.append(f"sendevent {dev} {EV_ABS} {ABS_X} {x0}")
    cmds.append(f"sendevent {dev} {EV_ABS} {ABS_Y} {y0}")
    cmds.append(f"sendevent {dev} {EV_ABS} {ABS_MT_TOUCH_MAJOR} 20")
    cmds.append(f"sendevent {dev} {EV_SYN} {SYN_REPORT} 0")
    
    # Loop path
    dt = duration / steps
    for i in range(1, steps + 1):
        theta = 2.0 * math.pi * (i / steps)
        x = int(cx + r * math.cos(theta))
        y = int(cy + r * math.sin(theta))
        cmds.append(f"sleep {dt:.4f}")
        cmds.append(f"sendevent {dev} {EV_ABS} {ABS_MT_POSITION_X} {x}")
        cmds.append(f"sendevent {dev} {EV_ABS} {ABS_MT_POSITION_Y} {y}")
        cmds.append(f"sendevent {dev} {EV_ABS} {ABS_X} {x}")
        cmds.append(f"sendevent {dev} {EV_ABS} {ABS_Y} {y}")
        cmds.append(f"sendevent {dev} {EV_SYN} {SYN_REPORT} 0")

    # Finger Up
    cmds.append(f"sendevent {dev} {EV_ABS} {ABS_MT_TRACKING_ID} -1")
    cmds.append(f"sendevent {dev} {EV_KEY} {BTN_TOUCH} 0")
    cmds.append(f"sendevent {dev} {EV_SYN} {SYN_REPORT} 0")

    sh_script = "\n".join(cmds)
    # Push to device and execute
    subprocess.run(["adb", "shell", f"sh -c '{sh_script}'"], check=True)
    print("Circle gesture dispatched successfully.")

if __name__ == "__main__":
    send_circle_gesture()
