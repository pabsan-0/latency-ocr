#! /usr/bin/env python3

import cv2
import pytesseract
import numpy as np
from PIL import ImageGrab
import argparse    
import os
import imutils
import mss
import random


# while true; do echo -e "$(date +%s%N)\n"; done
monitor = {"top": 0, "left": 0, "width":  1920, "height": 1080}
ROI_CACHE_FILE = "/tmp/latency_profiler__roi_cache.txt"
source_roi = []
sink_roi = []

def on_mouse_event(image):

    def callback(event, x, y, flags, param):
        global source_roi
        global sink_roi

        if event == cv2.EVENT_LBUTTONDOWN:
            source_roi = [(x, y)]
        elif event == cv2.EVENT_LBUTTONUP:
            source_roi.append((x, y))

        elif event == cv2.EVENT_MBUTTONDOWN:
            sink_roi = [(x, y)]
        elif event == cv2.EVENT_MBUTTONUP:
            sink_roi.append((x, y))
        
        elif event == cv2.EVENT_MOUSEMOVE:
            canvas = image.copy()
            if len(source_roi) == 1:
                canvas = cv2.rectangle(canvas, source_roi[0], (x, y), (0, 255, 0), 2)
            if len(source_roi) == 2:
                canvas = cv2.rectangle(canvas, source_roi[0], source_roi[1], (0, 255, 0), 2)
            if len(sink_roi) == 1:
                canvas = cv2.rectangle(canvas, sink_roi[0], (x, y), (255, 0, 0), 2)
            if len(sink_roi) == 2:
                canvas = cv2.rectangle(canvas, sink_roi[0], sink_roi[1], (255, 0, 0), 2)

            canvas = cv2.putText(canvas, '(LMB) Source: %r' % source_roi, (50, 50), 0, 1, (0,   0, 0), 9)
            canvas = cv2.putText(canvas, '(LMB) Source: %r' % source_roi, (50, 50), 0, 1, (0, 255, 0), 3)

            canvas = cv2.putText(canvas, '(MMB) Sink: %r' % sink_roi,     (50, 80), 0, 1, (  0, 0, 0), 9)
            canvas = cv2.putText(canvas, '(MMB) Sink: %r' % sink_roi,     (50, 80), 0, 1, (255, 0, 0), 3)

            cv2.imshow("image", canvas)

    return callback 


def select_rois():

    # Grab the whole screen
    # image = np.array(ImageGrab.grab())
    with mss.mss() as sct:
        image = np.array(sct.grab(monitor))

    # Run Roi defining window
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", on_mouse_event(image))
    cv2.imshow("image", image)
    while True:
        key = cv2.waitKey(6)
        if key == ord("q"):
            break
    cv2.destroyAllWindows()


def roi_cache_read(location=ROI_CACHE_FILE):
    arr = np.loadtxt(location)
    source = arr[:2].tolist()
    sink = arr[2:].tolist()
    print("read from %s" % location)
    return source, sink


def roi_cache_write(location=ROI_CACHE_FILE):
    np.savetxt(location, np.vstack((source_roi, sink_roi)))
    print("wrote to %s" % location)


def image_slice(image, xyxy):
    x1 = int(min(xyxy[0][0], xyxy[1][0]))
    y1 = int(min(xyxy[0][1], xyxy[1][1]))
    x2 = int(max(xyxy[0][0], xyxy[1][0]))
    y2 = int(max(xyxy[0][1], xyxy[1][1]))
    return image[y1:y2, x1:x2]
    

def infer_ocr(image):
    canvas = np.array(image.copy())
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    canvas = cv2.resize(canvas, (0, 0), fx = 4, fy = 4)
    canvas = imutils.resize(canvas, width=900)
    canvas = cv2.GaussianBlur(canvas, (5,5),0)
    _, canvas = cv2.threshold(canvas, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
     
    text = pytesseract.image_to_string(canvas, config=
        "-c tessedit"
        "_char_whitelist=' '0123456789"
        " --psm 6"
    )
    return text.strip(), canvas


def compute_latency(source, sink):
    try:
        return int(source) - int(sink)
    except Exception as e:
        return None



if __name__ == "__main__":

    # construct the argument parser and parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-rois", action="store_true", help="force-run bbox selection wizard")
    parser.add_argument("--preview", action="store_true", help="whether to preview pre-ocr snip")
    parser.add_argument("--sanity-check", type=float, default=0, help="chance to store snips for hand verification")
    # parser.add_argument("--rate", type=float, help="rate (hz)_at which to run OCR")
    args = parser.parse_args()
 
    
    if not os.path.isfile(ROI_CACHE_FILE) or args.new_rois:
        select_rois()
        roi_cache_write()
    else: 
        source_roi, sink_roi = roi_cache_read()
        
    while True:
        # screenshot = np.array(ImageGrab.grab())
        with mss.mss() as sct:
            screenshot = np.array(sct.grab(monitor))

        source_text, source_img  = infer_ocr(image_slice(screenshot, source_roi))
        sink_text, sink_img   = infer_ocr(image_slice(screenshot, sink_roi))
        latency = compute_latency(source_text, sink_text)
        
        if args.preview:
            cv2.imshow("preview", source_img)
            cv2.imshow("sink", sink_img)
            cv2.waitKey(5)

        if random.random() < args.sanity_check:
            cv2.imwrite("cache/%s_%s_%s.png" % (source_text, sink_text, latency),
                np.vstack((source_img, sink_img))
            )
            print("saved snips to cache!")

        print(source_text, sink_text, end=":\t")
        print(latency if latency else "")
