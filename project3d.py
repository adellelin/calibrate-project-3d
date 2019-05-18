from os.path import expanduser
from os.path import join
import json
import numpy as np
import cv2
import pyquaternion as pyq
import os

#xml
from feature_tools import get_face_bb, drawPoly, lm_bbox_to_cvpoints
import xml_helper as xhelp
import xml.etree.ElementTree as xmltree
import xml.dom.minidom as minidom
from lxml import etree

# camera coordinate text files
data_file = expanduser('~/Desktop/LAIKATEST/lionel/polar_frame_positions_tight_framing.txt')
# 3d labeled points
model_keypoints_json = expanduser('~/Desktop/LAIKATEST/lionel/lionel_downs.obj_ibug68.ljson')
# input frame directory
frame_directory = expanduser('~/Desktop/LAIKATEST/Lionel_cherry_set/001/')
#frame_directory = expanduser('~/Desktop/LAIKATEST/lionel/lionel_input/')
# frame names up to count digit
frame_prefix = 'dev.brdf_scan_lionel.tex1.001.L.'
# where to put the output images
output_directory = expanduser('~/Desktop/LAIKAOUTPUT/lionel_out/001')
# the xmlfile to be created
xmlfilename = expanduser('~/Desktop/LAIKATEST/lionel/project3d_001.xml')
# prefix for output images
output_image_prefix = 'training/'

#hand-tuned parameters to get keypoint positions from  3D model json

#model scale: how much you have to expand model coordinates (from arbitrary units to 3-D mm space)
model_scale = 2.5
#model offset: displacement between [0, 0, 0] of fixture and the origin in model space
model_offset = np.array([0.01, 0.61, -0.125])
#model quaternion: rotation needed to bring model from model space into fixture space -- requires 8 degree rotation for this lionel model
model_quaternion = pyq.Quaternion(axis=[1.0, 0.0, 0.0], angle=-8*np.pi/180.0)

# parse header -- reads the polar coordinate txt file
head_rotations = []
azimuths = []
elevations = []
radii = []
pans = []
tilts = []
indices = []

#TODO:  create new
xmlroot = xhelp.create_xml()
#xmlroot = xmlfilename
facePoints = np.empty(shape=(68, 2), dtype='int')

if os.path.exists(output_directory) is False:
    os.mkdir(output_directory)


# read frames in folder
frame_list = []
for root, dir, files in os.walk(frame_directory):
    for file in files:
        if '.DS_Store' in file:
            continue
        file_number = file.split('.')[-2]
        frame_list.append(int(file_number))

print('frame list', frame_list)

with open(data_file, mode='r') as f:
    # skip unused header data
    for _ in range(2):
        f.readline()

    # get focal length and sensor width from header
    lens_line = f.readline().split()
    focal_length = float(lens_line[2].split('m')[0])
    sensor_width = float(lens_line[4])

    # skip unused header data
    for _ in range(3):
        f.readline()

    d2r = np.pi / 180.

    for cur_line in f.readlines():
        split = cur_line.split()
        if (int(split[0])) in frame_list:
            head_rotations.append(d2r*float(split[1]))
            azimuths.append(d2r*float(split[2]))
            elevations.append(d2r*float(split[3]))
            radii.append(float(split[4]))
            pans.append(d2r*float(split[5]))
            tilts.append(d2r*float(split[6]))
            indices.append(int(split[0]))

print('selected indices', indices)
print('azimuth', len(azimuths))


# parse model keypoints loads the JSON output of the 3D model
with open(model_keypoints_json, mode='r') as f:
    loaded = json.load(f)
model_keypoints = np.array(loaded['landmarks']['points'])

#create a loop that cycles through every frame from one of the head rotation folders
#for each frame do the full projection of of all points of interest onto the frame and write the annotated frame to the output directory
num_frames = len(head_rotations)

#for frame_n in range(num_frames):
for i, frame_n in enumerate(indices):
    print(i, frame_n)
    azimuth = azimuths[i]
    elevation = elevations[i]
    rotation = head_rotations[i]
    radius = radii[i]

    width, height = 2348., 1566.  # pixel size of inputs
    focal_length = 105.9  # mm
    sensor_width = 36  # horizontal mm

    # fix these vals so that camera is always pointed at origin
    pan = -np.pi + azimuth  # acutally -180 - azimuth in files... why?
    tilt = -elevation  # always true in files -- in order for camera to alway point to fixture origin it must be equal to opposite of elevation

    # note negative sign in tilt angle: positive right handed rotation about x points z 'down',
    #   but positive tilt is defined as an 'up' rotation
    q_tilt = pyq.Quaternion(axis=[1., 0., 0.], angle=-tilt)
    q_pan = pyq.Quaternion(axis=[0., 1., 0.], angle=pan)

    # note negative sign in elevation angle: positive right handed rotation about x points z 'down',
    #   but positive elevation is defined as an 'up' rotation
    q_elev = pyq.Quaternion(axis=[1., 0., 0.], angle=-elevation)
    q_azim = pyq.Quaternion(axis=[0., 1., 0.], angle=azimuth)

    # define quaternion that rotates a vector from earth frame to camera frame
    #fixture is rotating relative to earth frame but no translation
    q_ce = (q_pan * q_tilt).inverse

    # define quaternion that rotates a vector from fixture frame to earth frame
    q_ef = pyq.Quaternion(axis=[0., 1., 0.], angle=rotation)

    # define the quaternion that defines the spherical coordinate rotation for determining camera position
    # getting from spherical coordinates to cartesian coordinates
    #q_trans is defining how you rotate the base translation vector for camera (when azimuth and elevation are both equal to zero) to actual translation
    q_trans = q_azim*q_elev
    # evaluate camera position in earth frame whic is the result of rotating from base position to actual camera position using the spherical coordinate rotation
    cam_pos_e = q_trans.rotate(np.array([0., 0., radius])) # this output should equal the cartesian values for camera position that Laika gave us

    # extrinsic matrix: fixture frame to camera frame
    ext = np.zeros((4, 4))
    ext[:3, :3] = (q_ce*q_ef).rotation_matrix  #rotation matrix that rotates from fixture coordinates to camera coordinates
    ext[:3, 3] = -q_ce.rotate(cam_pos_e) #translation from fixture to camera in camera coordinates

    # intrinsic matrix: camera frame to virtual image frame (has a negative multiplier)
    intrinsic = np.zeros((4, 4))
    intrinsic[0, 0] = -focal_length/sensor_width*width
    intrinsic[1, 1] = -focal_length/sensor_width*width
    intrinsic[2, 2] = 1
    intrinsic[0, 2] = width//2
    intrinsic[1, 2] = height//2

    # define composed projection matrix: earth frame to "virtual image" frame projected image to back of camera but flipped
    #left to right and top to bottom  so that it is in coordinates that that match our pixels
    #Virtual image space is still a 3-D space but now the x and y match our image
    #this multiplier is defined such that when you are z=1 in the virtual image space, then that equals pixel space in X, Y in the image
    proj = intrinsic.dot(ext)

    def pix(fixture_position):
        """
        map a point in fixture frame into pixel coordinates by projecting into virtual image frame and
            normalizing by z coordinate
        :param fixture_position: cartesian position in fixture frame (x, y, z, 1)
        :return: (x,y) position in pixel coordinates
        """
        virtual = proj.dot(fixture_position)
        return virtual[:2]/virtual[2]

    #input image frame
    img = cv2.imread(join(frame_directory, f'{frame_prefix}{frame_n:04d}.png'))
    print("image", f'{frame_prefix}{frame_n:04d}.png')
    #img = cv2.imread(join(frame_directory, f'{frame_prefix}{frame_n+1:04d}.png'))

    #origin in pixel coordinates at Z=1 of the fixture origin. This mis where fixture orini (0,0,0) maps to pixels
    origin = pix(np.array([0., 0., 0., 1.0]))
    # paint green dot in center of fixture
    cv2.circle(img, (int(origin[0]), int(origin[1])), 10, (0, 255, 0), cv2.FILLED)

    #Lines below are used to paint april tag coordinates
    pix_per_cm = 150.  # measured from cm bar in image
    #pixel x, y of upper left corner of apriul tag in relation to the front-facing first image of folder
    grid_x = 205.  # measured april tag origin in pixels from front image
    grid_y = 308.  # measured april tag origin in pixels from front image
    #measure points between april tags (left corner to left corner) and normalized by pix_per_cm
    grid_spacing = (373-187)/pix_per_cm  # measured april tag grid spacing from front image

    #infered position of upper left corner of april tag in fixture space
    x0 = (grid_x - width//2)/pix_per_cm
    y0 = (height//2 - grid_y)/pix_per_cm

    #paint a 6x4 array of circles on a grid in fixture space on a plane at Z= 0
    for n in range(0, 6):
        for m in range(0, 4):
            p1 = pix(np.array([x0+grid_spacing*m, y0-grid_spacing*n, 0.0, 1.0]))
            cv2.circle(img, (int(p1[0]), int(p1[1])), 10, (0, 255, 0), cv2.FILLED)

    #convert from model space to fixture space, then convert from fixture space to pixel space and paint keypoints
    #homogenous means we are appending a one to the 3-d array
    for kp_n in range(68):
        kp_homogenous = np.ones(4)
        #transformation from model space keypoints to fixture space keypoints
        kp_homogenous[:3] = model_scale*(model_quaternion.rotate(model_keypoints[kp_n])+model_offset)
        #the next line translates from fixture space to pixel space
        p1 = pix(kp_homogenous)

        facePoints[kp_n, :] = p1
        #now draw
        cv2.circle(img, (int(p1[0]), int(p1[1])), 8, (0, 255, 255), cv2.FILLED)

    # TODO: add to xml file
    pointsReshape = np.round(facePoints) #[:, np.newaxis, :]

    #TODO: get box
    # box_pt = np.array([0,1])
    # nbb_ul = box_pt.reshape(2,)
    # nbb_lr = nbb_ul

    # get bounding box around  all shape points
    nbb_ul, nbb_lr, is_implied = get_face_bb(img, pointsReshape)
    # convert bounding box open cv box format for visualizing
    cv_ul, cv_lr = lm_bbox_to_cvpoints(nbb_ul, nbb_lr)
    pts_filename = join(output_image_prefix, f'{frame_prefix}{frame_n:04d}.png')
    #pts_filename = f'{frame_prefix}{frame_n:04d}.png'
    print(pts_filename)
    drawPoly(img, pointsReshape, pts_filename, True, cv_ul[::-1], cv_lr[::-1])
    # flip the box x,y points, pts_filename == name of image file
    xmlroot = xhelp.append_xml(xmlroot, pointsReshape, pts_filename, nbb_ul[::-1], nbb_lr[::-1])

    #cv2.imwrite(pts_filename, img)

#TODO: write xml to file and format to human readable
xmlstring = xmltree.tostring(xmlroot, 'utf-8')
reparsed = minidom.parseString(xmlstring)
ptstree = reparsed.toprettyxml(indent="\t")
#print("xml", ptstree)
with open(xmlfilename, "w") as fh:
    fh.write(ptstree)

# make video from frames
# ffmpeg -r 30 -f image2 -i frame%04d.jpg -start_number 1 -vcodec libx264 -crf 25  -pix_fmt yuv420p test.mp4
