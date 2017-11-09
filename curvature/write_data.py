"""
Functions for creating and committing data
"""
import datetime

import numpy as np
import h5py
import scipy.optimize
import cv2
import imutils
from imutils import contours


def extract_radius_of_curvature(image_path, radius_scale):
    """
    Extract contours from image

    Parameters
    ----------
    image_path
    radius_scale
    """

    # Load image in grayscale, blur, and detect edges
    img = cv2.imread(image_path,0)
    blurred = cv2.GaussianBlur(img, (7,7), 0)
    edges = cv2.erode(cv2.dilate(cv2.GaussianBlur(cv2.Canny(blurred,20,100),(9,9),0), None, iterations=1), None, iterations=1)
    
    # find contours in the edge map
    cnts = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    (cnts, _) = contours.sort_contours(cnts)
    
    # loop over the contours individually 
    # NOTE: the last circle in the loop is the dot scale
    radius_pixel = []
    x_text = []
    y_text = []
    x_center = []
    y_center = []
    for c in cnts:
        # if the contour is not sufficiently large, ignore it
        if cv2.contourArea(c) < 190:
            continue
        # gets the x and y values and makes a list of them seperately
        x = np.array(c)[:,0,0]
        y = np.array(c)[:,0,1]
        # Get the center of each contour to use as the first guest to determine the center point of the circle
        M = cv2.moments(c)
        x_m = int(M["m10"] / M["m00"])
        x_text.append(x_m-100)
        y_m = int(M["m01"] / M["m00"])
        y_text.append(y_m)
        center_estimate = x_m, y_m
        
        # define a few functions for fitting
        def calc_R(xc, yc):
            # calculate the distance of each 2D points from the center (xc, yc)
            return np.sqrt((x-xc)**2 + (y-yc)**2)

        def f_2(c):
            # calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc)
            Ri = calc_R(*c)
            return Ri - Ri.mean()
        
        # use the x,y listed above and center_estimate to fit circle to arcs
        center_2, ier = scipy.optimize.leastsq(f_2, center_estimate)
        xc_2, yc_2 = center_2
        Ri_2 = calc_R(*center_2)
        # average all the radii to get mean radius for each
        R_2 = Ri_2.mean()
        radius_pixel.append(R_2)
        x_center.append(xc_2)
        y_center.append(yc_2)
        
    # convert to numpy array
    radius_pixel = np.array(radius_pixel)
    # Normalize the radii by the dot size
    radius_physical = radius_pixel*radius_scale/R_2
        
    return img, x_center, y_center, radius_physical, radius_scale/R_2


def commit_image(dbase_name, image_path, strain, radius_scale, metadata, date=None):
    # extract curvatures
    img, x_center, y_center, radius_physical, physical_scale = extract_radius_of_curvature(image_path, radius_scale)
    # separate LCE curves from scale curve
    x_center_scale = x_center.pop()
    y_center_scale = y_center.pop()
    radius_physical_scale = radius_physical[-1]
    radius_physical = radius_physical[:-1]
    # adjust metadata
    metadata['physical_conversion'] = physical_scale
    metadata['x_center_scale'] = x_center_scale
    metadata['y_center_scale'] = y_center_scale
    metadata['physical_radius_scale'] = radius_physical_scale
    # write to file
    if date is None:
        date = datetime.datetime.today()
    if not isinstance(date, datetime.datetime):
        date = datetime.datetime.strptime(date, '%m %d %Y')
    with h5py.File(dbase_name, 'a') as hf:
        grp_name = date.strftime('%m_%d_%y')
        # create day group if it does not already exist
        if grp_name not in hf:
            grp = hf.create_group(grp_name)
        else:
            grp = hf[grp_name]
        # create image subgroup
        subgrp_num = len([k for k in grp.keys()])
        subgrp = grp.create_group('{}'.format(subgrp_num))
        # add datasets to subgroup
        dset = subgrp.create_dataset('image', data=img)
        dset.attrs['filepath'] = image_path
        subgrp.create_dataset('strain', data=strain)
        subgrp.create_dataset('physical_radii', data=radius_physical)
        subgrp.create_dataset('x_center', data=np.array(x_center))
        subgrp.create_dataset('y_center', data=np.array(y_center))
        # add metadata
        for key in metadata:
            subgrp.attrs[key] = metadata[key]
            