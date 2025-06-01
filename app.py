import cv2
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.9,maxHands=2)

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
      # Hand 1
      hand1 = hands[0]
      lmList1 = hand1['lmList'] # Gives 21 Landmarks points
      bbox1 = hand1['bbox']  # Bounding Box info x,y,w,h
      center1 = hand1['center'] # Center of hand ie cx,cy
      handType1 = hand1['type'] # Type of Hand
      fingers1 = detector.fingersUp(hand1)

      # Hand 2
      if len(hands) == 2:
          hand2 = hands[1]
          lmList2 = hand2['lmList']  # Gives 21 Landmarks points
          bbox2 = hand2['bbox']  # Bounding Box info x,y,w,h
          center2 = hand2['center']  # Center of hand ie cx,cy
          handType2 = hand2['type']  # Type of Hand
          fingers2 = detector.fingersUp(hand2)

          length, info, img = detector.findDistance(center1,center2,img,scale=9)
          cv2.putText(img,str(int(length)),(25,45),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,255),2)

    cv2.imshow("Camera Feed",img)
    cv2.waitKey(1)