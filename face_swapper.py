import dlib
import cv2
import numpy as np

import models
import NonLinearLeastSquares
import ImageProcessing

from drawing import *

import FaceRendering
import utils

predictor_path = "./shape_predictor_68_face_landmarks.dat"

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
mean3DShape, blendshapes, mesh, idxs3D, idxs2D = utils.load3DFaceModel("./candide.npz")

projectionModel = models.OrthographicProjectionBlendshapes(blendshapes.shape[0])

modelParams = None
lockedTranslation = False
drawOverlay = False
cap = cv2.VideoCapture(0)
writer = None

textureImg1 = None
textureCoords1 = None

print "-----"
print "Press SPACE to scan the first face."
print "-----"

while textureCoords1 is None:
    cameraImg = cap.read()[1]
    cv2.imshow('Face Swapper', cameraImg)

    key = cv2.waitKey(1)

    if key % 256 == 32:
        try:
            textureCoords1 = utils.getFaceTextureCoords(cameraImg, mean3DShape, blendshapes, idxs2D, idxs3D, detector, predictor)
            textureImg1 = cameraImg
            print "First face scanned."
        except:
            print "Please try again."
            pass

textureImg2 = None
textureCoords2 = None

print "-----"
print "Press SPACE to scan the second face."
print "-----"

while textureCoords2 is None:
    cameraImg = cap.read()[1]
    cv2.imshow('Face Swapper', cameraImg)

    key = cv2.waitKey(1)

    if key % 256 == 32:
        try:
            textureCoords2 = utils.getFaceTextureCoords(cameraImg, mean3DShape, blendshapes, idxs2D, idxs3D, detector, predictor)
            textureImg2 = cameraImg
            print "Second face scanned."
        except:
            print "Please try again."
            pass

renderer = FaceRendering.FaceRenderer(cameraImg, mesh)
renderer.setFirstFaceImageAndCoordinates(textureImg1, textureCoords1)
renderer.setSecondFaceImageAndCoordinates(textureImg2, textureCoords2)

while True:
    cameraImg = cap.read()[1]
    shapes2D = utils.getFaceKeypoints(cameraImg, detector, predictor, 320)

    if shapes2D is not None and len(shapes2D) == 2:
        avg = sum(shapes2D[0][0]) / len(shapes2D[0][0])

        if avg > 250:
            leftFace = shapes2D[0]
            rightFace = shapes2D[1]
        else:
            leftFace = shapes2D[1]
            rightFace = shapes2D[0]

        for index, shape2D in enumerate([leftFace, rightFace]):
            modelParams = projectionModel.getInitialParameters(mean3DShape[:, idxs3D], shape2D[:, idxs2D])

            modelParams = NonLinearLeastSquares.GaussNewton(modelParams, projectionModel.residual, projectionModel.jacobian, ([mean3DShape[:, idxs3D], blendshapes[:, :, idxs3D]], shape2D[:, idxs2D]), verbose=0)

            shape3D = utils.getShape3D(mean3DShape, blendshapes, modelParams)
            renderedImg = renderer.render(shape3D, index)

            mask = np.copy(renderedImg[:, :, 0])
            renderedImg = ImageProcessing.colorTransfer(cameraImg, renderedImg, mask)
            cameraImg = ImageProcessing.blendImages(renderedImg, cameraImg, mask)

            if drawOverlay:
                drawPoints(cameraImg, shape2D.T)
                drawProjectedShape(cameraImg, [mean3DShape, blendshapes], projectionModel, mesh, modelParams, lockedTranslation)

    if writer is not None:
        writer.write(cameraImg)

    cv2.imshow('Face Swapper', cameraImg)
    key = cv2.waitKey(1)

    if key == 27:
        break
    if key == ord('t'):
        drawOverlay = not drawOverlay
