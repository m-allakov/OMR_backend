import cv2 as cv
import numpy as np
import utilis
###################
path="1.png"
widthImg=500
heightImg=500
qestions = 5  
choises = 5
ans = [1,2,0,3,4] # her soru için cevap
webCamFeed = True
cameraNo = 0
###################
cam = cv.VideoCapture(cameraNo)
cam.set(10,150)
img = cv.imread(path)
# while True:
#     if webCamFeed: 
#         syccess, img = cam.read()
#     else:
#         img = cv.imread(path)

    
    #PREPROCESING
img= cv.resize(img,(widthImg, heightImg))
imgContours = img.copy()
imgFinally = img.copy()
imgBiggestContours = img.copy()
imgGray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
imgBlur=cv.GaussianBlur(imgGray,(5,5),1)
imgCanny=cv.Canny(imgBlur,10,50)

   
#TUM KONTURLARI BULMAK
contours , hierarcy = cv.findContours(imgCanny, cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
cv.drawContours(imgContours, contours, -1, (255,0,0), 5)

# TUM DIKTORTKENLERI BULMA
rectCon = utilis.rectContours(contours)
biggestContour = utilis.getContourPoints(rectCon[0])
gradePoints = utilis.getContourPoints(rectCon[1])
#print(biggestContour)

if biggestContour.size != 0 and gradePoints.size !=  0: 
    cv.drawContours(imgBiggestContours,biggestContour, -1,(0,0,255),10)
    cv.drawContours(imgBiggestContours,gradePoints, -1,(0,255,0),10)
                
    biggestContour = utilis.reOrder(biggestContour)
    gradePoints = utilis.reOrder(gradePoints)

    pt1 = np.float32(biggestContour)
    pt2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    matrix = cv.getPerspectiveTransform(pt1, pt2)
    imgWarpedColred = cv.warpPerspective(img, matrix, (widthImg, heightImg))

    ptG1 = np.float32(gradePoints)
    ptG2 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])
    matrixG = cv.getPerspectiveTransform(ptG1, ptG2)
    imgGradeDisplay = cv.warpPerspective(img, matrixG, (325, 150))

    # EŞIK UYGULAMAK
    # imgWarpGray = cv.cvtColor(imgWarpedColred,cv.COLOR_BGR2BGRA)
    imgWarpGray = cv.cvtColor(imgWarpedColred, cv.COLOR_BGR2GRAY)
    # print('------',imgWarpGray.shape)
    imgThresh = cv.threshold(imgWarpGray,170,255,cv.THRESH_BINARY_INV)[1]
                


    boxes = utilis.splitBoxes(imgThresh)
    #cv.imshow('boxes', boxes[1]) 
    #print(cv.countNonZero(boxes[1]),cv.countNonZero(boxes[2]))

    #! GETTING PIXEL VALUES BOXES
    myPixelValue = np.zeros((qestions,choises))
    countC=0
    countR=0

    for image in boxes:
        totalPixels = cv.countNonZero(image)
        myPixelValue[countR][countC] = totalPixels
        countC += 1
        if (countC == choises): countR +=1 ; countC=0
    #print(myPixelValue)    

    # ************** SORTING THE QUESTIONS BY NUMBER OF ANSWERS *******************
    myIndex = []
    for x  in range(0,qestions):
        arr = myPixelValue[x]
        #print('arr',arr)
        myIndexVal = np.where(arr==np.amax(arr))
        #print(myIndexVal[0])
        myIndex.append(myIndexVal[0][0])
    #print(myIndex)
                    
    # ***** GRADING *******
    gradding = []
    for x in range(0,qestions):
                    if ans[x] == myIndex[x]:
                        gradding.append(1)
                    else: gradding.append(0)
    #print(gradding)
    score = (sum(gradding)/qestions) *100 #FINAL SCORE
    #print(score)



    # DISPLAY ANSWERS
    imgResult = imgWarpedColred.copy()
    imgResult = utilis.showAnswers(imgResult, myIndex, gradding, ans, qestions, choises)
    imgRowOrawning = np.zeros_like(imgWarpedColred)
    imgRowOrawning = utilis.showAnswers(imgRowOrawning, myIndex, gradding, ans, qestions, choises)
    Invmatrix = cv.getPerspectiveTransform(pt2, pt1)
    imgInvWarpedColred = cv.warpPerspective(imgRowOrawning, Invmatrix, (widthImg, heightImg))

    imgRowGrade = np.zeros_like(imgGradeDisplay)
    cv.putText(imgRowGrade, str(int(score))+'%',(65, 100), cv.FONT_HERSHEY_COMPLEX,3,(255,255,255),3)
    invMatrixG = cv.getPerspectiveTransform(ptG2, ptG1)
    imgInvGradeDisplay = cv.warpPerspective(imgRowGrade, invMatrixG, (widthImg, heightImg))


    imgFinally = cv.addWeighted(imgFinally,1, imgInvWarpedColred,1,0)
    imgFinally = cv.addWeighted(imgFinally,1, imgInvGradeDisplay,1,0)
    cv.imshow('windowName', imgFinally)
            





    imgBlank = np.zeros_like(img)
    imageArray=([img,imgGray,imgBlur,imgCanny],
            [imgContours,imgBiggestContours,imgWarpedColred,imgThresh],
            [imgResult,imgRowOrawning,imgInvWarpedColred,imgFinally]
            )
        
imgStacked = utilis.stackImages(0.5,imageArray)



#cv.imshow('Trashed image', imgThresh)
#cv.imshow('original', img)
#cv.imshow("Result", imgWarpedGradeDisplay)

cv.imshow('Stacked Images', imgStacked)
cv.waitKey(0)

# if cv.waitKey(1) & 0xFF == ord('s'):
#     cv.imwrite("result.jpg", imgFinally)
#     cv.waitKey(300)