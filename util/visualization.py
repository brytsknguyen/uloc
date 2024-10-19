import matplotlib.pyplot as plt
import numpy as np
from .eval import Result
import sys
sys.path.append("../")
from PIL import Image
import pickle
from mpl_toolkits.axes_grid1 import make_axes_locatable

def visualize_sample(train_dataset,test_dataset,Yscaler,half=0):
    
    Xtrain, Ytrain = train_dataset.__getitem__(range(train_dataset.__len__()))
    Xtest, Ytest = test_dataset.__getitem__(range(test_dataset.__len__()))

    Ytrain_unscaled = Yscaler.inverse_transform(Ytrain[:, -1, :])
    Ytest_unscaled = Yscaler.inverse_transform(Ytest[:, -1, :])

    if half:
        xtest, ytest = Ytest_unscaled[:, 0] , Ytest_unscaled[:, 1] 
        xtrain, ytrain = Ytrain_unscaled[:, 0] , Ytrain_unscaled[:, 1] 
    else:
        xtest, ytest = (Ytest_unscaled[:, 0] + Ytest_unscaled[:, 3])/2, (Ytest_unscaled[:, 1] + Ytest_unscaled[:, 4])/2
        xtrain, ytrain = (Ytrain_unscaled[:, 0] + Ytrain_unscaled[:, 3])/2, (Ytrain_unscaled[:, 1] + Ytrain_unscaled[:, 4])/2

    figsize=(8,8)
    fig, axes = plt.subplots(1, 1, figsize=figsize)

    axes.plot(xtrain, ytrain, '.r', markersize=0.5, label='Training Set')
    axes.plot(xtest, ytest, '.b', markersize=0.5, label='Test Set')

    axes.set_xlabel('X')
    axes.set_ylabel('Y')
    axes.set_title('Left')
    axes.legend()

    axes.set_aspect('equal')

    plt.tight_layout()
    plt.show()



transformation_matrix = np.array([
    [0.7546453093136192, 0.6557852595445468, 0.02135768000144952, 48.71525562352759],
    [-0.655183740953685, 0.7549045314583982, -0.029213249964888575, 107.02710927458888],
    [-0.035280628124896816, 0.00805243737436802, 0.9993450032553546, 9.056978426429225],
    [0.0, 0.0, 0.0, 1.0]
])

def transform_coordinates(input_array, transformation_matrix,train_one_anchor):
    n = input_array.shape[0]
    if train_one_anchor:
        points_1 = input_array[:, :3]  
        points_1_homogeneous = np.hstack((points_1, np.ones((n, 1))))  
        new_points_1 = np.dot(points_1_homogeneous, transformation_matrix.T)[:, :3]  

        transformed_array = new_points_1
    else:
        points_1 = input_array[:, :3]  
        points_2 = input_array[:, 3:] 
        points_1_homogeneous = np.hstack((points_1, np.ones((n, 1))))  
        points_2_homogeneous = np.hstack((points_2, np.ones((n, 1))))  

        new_points_1 = np.dot(points_1_homogeneous, transformation_matrix.T)[:, :3]  
        new_points_2 = np.dot(points_2_homogeneous, transformation_matrix.T)[:, :3]  

        transformed_array = np.hstack((new_points_1, new_points_2))       
    
    return transformed_array


def plotPathError(Y, Ypred, ax,colorScalePiv, train_on_slam, train_one_anchor, trial=''):

    # Calculate and scale the error
    res = Result(train_one_anchor)
    res.evaluate(Ypred, Y)
    err = np.linalg.norm(res.abs_diff, axis=1)
    err_clamped = err.copy()
    err_clamped[err > colorScalePiv] = colorScalePiv

    figsize = (12, 12)
    plt.rcParams.update({'font.size': 16})

    # Plot the scaled color map

    # Load the image and dimensions
    img = Image.open('materials/ntu_2d_rgb.png')
    xylim = np.load('materials/ntu_2d_rgb_xylim.npy')
    # Resize the image
    xspan = (xylim[0, 1] - xylim[0, 0])
    yspan = (xylim[1, 1] - xylim[1, 0])
    img = img.resize((int(yspan), int(xspan)))
    # Load the image to the graph
    x_shift = xylim[0, 0]
    y_shift = xylim[1, 0]
    ax.imshow(img, extent=[x_shift, img.size[1] + x_shift, y_shift, img.size[0] + y_shift], alpha=0.5)

    if train_on_slam:
        Y     = transform_coordinates(Y,transformation_matrix,train_one_anchor)
        Ypred = transform_coordinates(Ypred,transformation_matrix,train_one_anchor)

    # Plot the path
    scatter = ax.scatter(Y[1, 0], Y[1, 1], c=err_clamped[1], cmap='jet', s=10, alpha = 0.8, label='ATV path')
    scatter = ax.scatter(Y[:, 0], Y[:, 1], c=err_clamped, cmap='jet', s=2, alpha = 0.8, label='')

    # Add a color bar to show the error
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.2)
    colorbar = plt.colorbar(scatter, label='Inference Error', cax=cax)  # Add a colorbar with a label

    # Plot the anchors
    with open('materials/anchor_pos.pkl', 'rb') as f:
        anchor_pos = pickle.load(f)

    anc_pos = np.array(list(anchor_pos['ntu_world'].values()))
    ax.scatter(anc_pos[:, 0], anc_pos[:, 1], c='r', s=50, label='anchor')
    
    ax.text(x=200, y=30, s=f'Trial: '+trial)
    ax.text(x=200, y= 5, s=f'RMSE: {res.rmse:.3} m')

    # Configure some visuals
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')

    ax.set_xlim([-150, 400])
    ax.set_ylim([-250, 150])
    ax.set_aspect('equal')
    ax.grid('on')
    ax.legend()

    return Y, Ypred
