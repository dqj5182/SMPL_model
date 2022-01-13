import os
import sys
sys.path.insert(0,'./smplpytorch')

import argparse
import numpy as np
import os.path as osp
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

from smpl import SMPL
import math
import cv2

mesh_model = SMPL()

def get_model(root_pose, pose, shape):
    pose = np.concatenate([root_pose, pose], axis=0)

    smpl_pose = torch.FloatTensor(pose).view(1, -1)
    smpl_shape = torch.FloatTensor(shape).view(1, -1)
    smpl_mesh_coord, smpl_joint_coord = mesh_model.layer['neutral'](smpl_pose, smpl_shape)
    smpl_mesh_coord = smpl_mesh_coord.numpy().astype(np.float32).reshape(-1, 3) * 1000
    return smpl_mesh_coord

def save_obj(v, f, file_name):
    obj_file = open(file_name, 'w')
    for i in range(len(v)):
        obj_file.write('v ' + str(v[i][0]) + ' ' + str(v[i][1]) + ' ' + str(v[i][2]) + '\n')
    for i in range(len(f)):
        obj_file.write('f ' + str(f[i][0]+1) + '/' + str(f[i][0]+1) + ' ' + str(f[i][1]+1) + '/' + str(f[i][1]+1) + ' ' + str(f[i][2]+1) + '/' + str(f[i][2]+1) + '\n')
    obj_file.close()

def rotation_matrix_to_rotVec(Rmat):    
    theta = math.acos(((Rmat[0, 0] + Rmat[1, 1] + Rmat[2, 2]) - 1) / 2)
    sin_theta = math.sin(theta)
    if sin_theta == 0:
        rx, ry, rz = 0.0, 0.0, 0.0
    else:
        multi = 1 / (2 * math.sin(theta))
        rx = multi * (Rmat[2, 1] - Rmat[1, 2]) * theta
        ry = multi * (Rmat[0, 2] - Rmat[2, 0]) * theta
        rz = multi * (Rmat[1, 0] - Rmat[0, 1]) * theta
    return np.array([rx, ry, rz])


def euler_to_rotMat(yaw, pitch, roll):
    Rz_yaw = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [          0,            0, 1]])
    Ry_pitch = np.array([
        [ np.cos(pitch), 0, np.sin(pitch)],
        [             0, 1,             0],
        [-np.sin(pitch), 0, np.cos(pitch)]])
    Rx_roll = np.array([
        [1,            0,             0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]])
    # R = RzRyRx
    rotMat = np.dot(Rz_yaw, np.dot(Ry_pitch, Rx_roll))
    return rotMat

def axis_angle_to_euler_angle(pose):
    euler_angles = []
    for angle in pose:
        rotation_matrix = cv2.Rodrigues(angle)[0]
        euler = rotationMatrixToEulerAngles(rotation_matrix)
        rotation_matrix2 = euler_to_rotMat(euler[2], euler[1], euler[0])

        if (rotation_matrix - rotation_matrix2).sum() > 0.1:
            assert 0
        
        euler = euler* 180 / math.pi
        euler_angles.append(euler)
    return np.stack(euler_angles)

def euler_angle_to_axis_angle(euler):
    euler = euler / 180 * math.pi
    rotation_matrix = euler_to_rotMat(euler[2], euler[1], euler[0]) 
    ddd = cv2.Rodrigues(rotation_matrix)
    axis = rotation_matrix_to_rotVec(rotation_matrix)

    return axis

SMPL_JOINT_NAMES = ('L_Hip', 'R_Hip', 'Torso', 'L_Knee', 'R_Knee', 'Spine', 'L_Ankle', 'R_Ankle', 'Chest', 'L_Toe',
        'R_Toe', 'Neck', 'L_Thorax', 'R_Thorax', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist',
        'R_Wrist', 'L_Hand', 'R_Hand')

DIRECTION = ('BEND1', 'BEND2', 'TWIST')


if __name__ == '__main__':
    root_pose = np.zeros((1,3))
    pose = np.zeros((23,3))
    shape = np.zeros((1,10))    
    

    pose2 = np.zeros((23,3))
    for i in range(len(pose)): pose2[i] = euler_angle_to_axis_angle(pose[i])

    mesh_cam = get_model(root_pose, pose2, shape)
    save_obj(mesh_cam, mesh_model.face, 'output.obj')
    
    print()
    print("-- SAVED MODEL --")

