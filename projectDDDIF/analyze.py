import os
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from skimage.transform import resize
import numpy as np
import cv2
# import cmapy
from pydicom import dcmread
from pydicom.uid import ExplicitVRLittleEndian
from captum.attr import GradientShap, DeepLift, DeepLiftShap, IntegratedGradients, GuidedGradCam, NoiseTunnel, Saliency, GuidedBackprop


def to_0_255(x):
    return (x-x.min())/(x.max()-x.min())*255


def write_dcm(raw, x, path):
    # Requires x of shape  (t,row,col,3)
    x = to_0_255(x)
    x = x.astype('uint8')
    raw.NumberOfFrames = x.shape[0]
    raw.Rows = x.shape[1]
    raw.Columns = x.shape[2]
    raw.PixelData = x.tobytes()
    raw.save_as(path)


def show_save_mov(video, save_path, file_type='mp4', norm=False, boundary=None, gray2color=None, fps=5, show=False, insert_text=None):
    if norm:
        if boundary is not None:
            video[video > boundary[0]] = boundary[0]
            video[video < boundary[1]] = boundary[1]
        video = ((video - np.min(video)) / (np.max(video) - np.min(video))) * 255
    video = np.asarray(video, dtype='uint8')

    frame_delay = int(1000 / fps)
    if file_type == 'mp4':
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    elif file_type == 'avi':
        fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    save_path = save_path + '.' + file_type
    out = cv2.VideoWriter(save_path, fourcc, fps, (video.shape[2],video.shape[1]))
    for frame in video:
        if gray2color is not None:
            frame = cv2.applyColorMap(frame, gray2color)
        if insert_text is not None:
            cv2.putText(frame, insert_text, (2, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            cv2.putText(frame, 'ILV'+' '*21+'AVR', (2, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        out.write(frame)
        if show:
            cv2.imshow('frame', frame)
            key = cv2.waitKey(frame_delay)

    out.release()
    cv2.destroyAllWindows()


def get_analyzer(methodID, model, if_smoothGrad):
    assert methodID in range(5)
    if methodID == 0:
        analyzer = Saliency(model)
        methodname = '_Saliency'
    if methodID == 1:
        analyzer = DeepLift(model)
        methodname = '_DL'
    if methodID == 2:
        analyzer = DeepLiftShap(model)
        methodname = '_DLshap'
    if methodID == 3:
        analyzer = GuidedBackprop(model)
        methodname = '_GB'
    if methodID == 4:
        analyzer = GuidedGradCam(model, model.layer4)
        methodname = '_GradCAM'

    if if_smoothGrad:
        analyzer = NoiseTunnel(analyzer)
        methodname = methodname+'smo'

    return analyzer, methodname


def run_analyze(analyzer, inputs, target):
    return analyzer.attribute(inputs=inputs, target=target, baselines=inputs*0)


def post_process(attributions, threshold):
    """Post-process the generated attributions"""
    assert threshold in ['abs', 'pos']
    if threshold == 'abs':
        attributions = abs(attributions)
    elif threshold == 'pos':
        attributions[attributions<0] = 0

    attributions = attributions.cpu().detach().numpy()[0, 0, ...]  # remove batch & channel dimension -> [t,x,y]
    attributions = np.uint8(to_0_255(attributions))
    attributions_color = []
    for i, att in enumerate(attributions):
        # att = cv2.applyColorMap(att, cv2.COLORMAP_JET)      #After this step the shape changes from (112,112) to (112,112,3)
        att = cv2.applyColorMap(att, cv2.COLORMAP_HOT)
        attributions_color.append(att)
    attributions_color = np.stack(attributions_color, axis=0)
    assert attributions_color.shape == (30, 112, 112, 3)
    return attributions_color


def analyze(X, target_classes, model, methodID, save_dir=None, file_name=None, tail='',
            save_vid_type='mp4', save_att_vid=True, save_input_vid=False, save_render_vid=False, save_render_npy=False,
            save_dcm=False, save_figs=False, threshold='pos', if_smoothGrad=False):
    os.makedirs(save_dir, exist_ok=True)

    # First, process and save the input X if needed
    if save_input_vid | save_render_vid | save_render_npy | save_figs:        # Then we would need X
        Xrgb = to_0_255(X.cpu().detach().numpy()[0, 0, ...])    # (b,c,t,x,y) -> (t,x,y)
        Xrgb = np.stack([Xrgb] * 3, axis=3)
    if save_input_vid:
        show_save_mov(video=Xrgb, save_path=save_dir + file_name, file_type=save_vid_type)

    # Second, run analyze and save if needed
    for c in target_classes:
        classname = '_class'+str(c)
        analyzer, methodname = get_analyzer(methodID, model, if_smoothGrad)
        attributions = run_analyze(analyzer, X, c)
        attributions_color = post_process(attributions, threshold)

        if save_render_vid | save_render_npy | save_figs:                         # Then we would need "render"
            render = attributions_color * 0.7 + Xrgb * 0.3

        if save_att_vid:
            show_save_mov(video=attributions_color, save_path=save_dir+file_name+tail+methodname+classname, file_type=save_vid_type)
        if save_render_vid:
            show_save_mov(nvideo=render, save_path=save_dir+file_name+tail+methodname+classname+'_overlay', file_type=save_vid_type)
        if save_render_npy:
            np.save(save_dir+file_name+tail+methodname+classname+'_overlay.npy', render)
        if save_figs:
            for i, (img, att, rnd) in enumerate(zip(Xrgb, attributions_color, render)):
                cv2.imwrite(save_dir + file_name + '_' + str(i) + '.png', img)
                cv2.imwrite(save_dir + file_name+tail+methodname+classname+'_heatmap_'+str(i)+'.png', att)
                cv2.imwrite(save_dir + file_name+tail+methodname+classname+'_render_'+str(i)+'.png', rnd)
