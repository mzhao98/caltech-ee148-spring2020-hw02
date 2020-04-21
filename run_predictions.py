import os
import numpy as np
import json
from PIL import Image
from numpy import unravel_index
from red_light_detector import *
import matplotlib.pyplot as plt

def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', 0)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value


def convolve_one_layer(I_layer, T_layer, stride):
    (n_rows,n_cols) = np.shape(I_layer)
    (t_rows,t_cols) = np.shape(T_layer)
    heatmap = np.zeros((n_rows,n_cols))

    rows_pad = int(t_rows/2)+1
    cols_pad = int(t_cols/2)+1

    T_layer_norm = (T_layer - np.mean(T_layer))/np.std(T_layer)
    # T_layer_norm = (T_layer_norm - np.min(T_layer_norm))/(np.max(T_layer_norm)-np.min(T_layer_norm))

    # zero pad image
    pad_size = max(rows_pad, cols_pad)
    I_layer_norm = (I_layer - np.mean(I_layer))/np.std(I_layer)
    # I_layer_norm = (I_layer_norm - np.min(I_layer_norm))/(np.max(I_layer_norm)-np.min(I_layer_norm))
    zero_padded_I = np.pad(I_layer_norm, pad_size, pad_with)

    adjust_start = pad_size-1
    adjust_row_end = n_rows + pad_size-1
    adjust_col_end = n_cols + pad_size-1

    rows_back = int(t_rows/2)
    rows_forward = t_rows - rows_back
    cols_back = int(t_cols/2)
    cols_forward = t_cols - cols_back

    total_divide = np.sum(np.ones((T_layer_norm.shape)))

    for r in range(adjust_start, adjust_row_end, stride):
        for c in range(adjust_start, adjust_col_end, stride):

            partial_img = zero_padded_I[r-rows_back:r+rows_forward, c-cols_back:c+cols_forward]
            # partial_img_norm = (partial_img - np.mean(partial_img))
            # partial_img_norm = (partial_img_norm - np.min(partial_img_norm))/(np.max(partial_img_norm)-np.min(partial_img_norm))

            conv_img = np.multiply(partial_img, T_layer_norm)

            total = np.sum(conv_img)
            heatmap[r-adjust_start, c-adjust_start] = total

    return heatmap, rows_back, rows_forward, cols_back, cols_forward





def compute_convolution(I, T, stride=None):
    '''
    This function takes an image <I> and a template <T> (both numpy arrays) 
    and returns a heatmap where each grid represents the output produced by 
    convolution at each location. You can add optional parameters (e.g. stride, 
    window_size, padding) to create additional functionality. 
    '''
    (n_rows,n_cols,n_channels) = np.shape(I)
    (k_rows,k_cols,k_channels) = np.shape(T)

    '''
    BEGIN YOUR CODE
    '''
    # heatmap = np.zeros((n_rows, n_cols))
    if stride is None:
        stride_size=1
    else:
        stride_size = stride
    # convolve each layer
    red_convolve, rows_back, rows_forward, cols_back, cols_forward = convolve_one_layer(I[:,:,0], T[:,:,0], stride=stride_size)
    green_convolve, _, _, _, _ = convolve_one_layer(I[:,:,1], T[:,:,1], stride=stride_size)
    blue_convolve, _, _, _, _ = convolve_one_layer(I[:,:,2], T[:,:,2], stride=stride_size)

    heatmap = np.array(red_convolve + green_convolve + blue_convolve)
    heatmap = heatmap/np.max(heatmap)


    '''
    END YOUR CODE
    '''

    return heatmap, rows_back, rows_forward, cols_back, cols_forward


def predict_boxes(heatmap, rows_back, rows_forward, cols_back, cols_forward):
    '''
    This function takes heatmap and returns the bounding boxes and associated
    confidence scores.
    '''

    output = []
    threshold = 0.7
    all_confidences = []
    rl_locations = []
    '''
    BEGIN YOUR CODE
    '''
    nrows = heatmap.shape[0]
    ncols = heatmap.shape[1]

    for r in range(nrows):
        for c in range(ncols):
            if heatmap[r, c] > threshold:
                # output.append([r-rows_back, c-cols_back, 
                #     r+rows_forward, c+cols_forward, heatmap[r, c]])
                all_confidences.append(heatmap[r, c])
                rl_locations.append([r,c])

    if len(output)==0:
        (max_row, max_col) = unravel_index(heatmap.argmax(), heatmap.shape)

        # output.append([max_row-rows_back, max_col-cols_back, 
        #     max_row+rows_forward, max_col+cols_forward, 1])
        all_confidences.append(heatmap[r, c])
        rl_locations.append([r,c])


    rl_locations = cluster_rl_locs(rl_locations)
    for [r, c] in rl_locations:
        output.append([r-rows_back, c-cols_back, 
            r+rows_forward, c+cols_forward, heatmap[r, c]])
    
    '''
    As an example, here's code that generates between 1 and 5 random boxes
    of fixed size and returns the results in the proper format.
    '''

    # box_height = 8
    # box_width = 6

    # num_boxes = np.random.randint(1,5)

    # for i in range(num_boxes):
    #     (n_rows,n_cols,n_channels) = np.shape(I)

    #     tl_row = np.random.randint(n_rows - box_height)
    #     tl_col = np.random.randint(n_cols - box_width)
    #     br_row = tl_row + box_height
    #     br_col = tl_col + box_width

    #     score = np.random.random()

    #     output.append([tl_row,tl_col,br_row,br_col, score])

    '''
    END YOUR CODE
    '''
    print("OUTPUT = ", output)
    return output, all_confidences


def detect_red_light_mf(I, init_redlightkernel, small_kernel, count=0, plot_heatmap=False):
    '''
    This function takes a numpy array <I> and returns a list <output>.
    The length of <output> is the number of bounding boxes predicted for <I>. 
    Each entry of <output> is a list <[row_TL,col_TL,row_BR,col_BR,score]>. 
    The first four entries are four integers specifying a bounding box 
    (the row and column index of the top left corner and the row and column 
    index of the bottom right corner).
    <score> is a confidence score ranging from 0 to 1. 

    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''

    '''
    BEGIN YOUR CODE
    '''
    # template_height = 8
    # template_width = 6

    # # You may use multiple stages and combine the results
    # T = np.random.random((template_height, template_width))
    img = I
    MAX_ROWS = img.shape[0]
    MAX_COLUMNS = img.shape[1]
    T1 = init_redlightkernel
    T2 = small_kernel

    # heatmap_combined, rows_back, rows_forward, cols_back, cols_forward = compute_convolution(I, T1)

    heatmap1, rows_back1, rows_forward1, cols_back1, cols_forward1 = compute_convolution(I, T1)
    heatmap2, rows_back2, rows_forward2, cols_back2, cols_forward2 = compute_convolution(I, T2)

    if plot_heatmap:
        plt.imshow(np.array(heatmap1), cmap='viridis', interpolation='nearest')
        plt.savefig('heatmap1_'+str(count)+'.png')
        plt.close()
        plt.imshow(np.array(heatmap2), cmap='viridis', interpolation='nearest')
        plt.savefig('heatmap2_'+str(count)+'.png')
        plt.close()
        print("done plotting")
    # heatmap_combined = (heatmap1+heatmap2)/2

    # if np.max(heatmap1) > np.max(heatmap2):
    #     rows_back, rows_forward, cols_back, cols_forward = rows_back1, rows_forward1, cols_back1, cols_forward1
    # else:
    #     rows_back, rows_forward, cols_back, cols_forward = rows_back2, rows_forward2, cols_back2, cols_forward2

    output1, all_confidences1 = predict_boxes(heatmap1, rows_back1, rows_forward1, cols_back1, cols_forward1)
    output2, all_confidences2 = predict_boxes(heatmap2, rows_back2, rows_forward2, cols_back2, cols_forward2)
    output1.extend(output2)


    # if max(all_confidences) < 0.8:
    #     print("CHECK SMALL KERNEL")
    '''
    END YOUR CODE
    '''

    # for i in range(len(output)):
    #     assert len(output[i]) == 5
    #     assert (output[i][4] >= 0.0) and (output[i][4] <= 1.0)

    return output1

# Note that you are not allowed to use test data for training.
# set the path to the downloaded data:
data_path = '../CS148/RedLights2011_Medium'

# load splits: 
split_path = '../data/hw02_splits'
if not os.path.exists(split_path):
    os.makedirs(split_path)
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# set a path for saving predictions:
preds_path = '../data/hw02_preds'
if not os.path.exists(preds_path):
    os.makedirs(preds_path)
os.makedirs(preds_path, exist_ok=True) # create directory if needed

# Set this parameter to True when you're done with algorithm development:
# done_tweaking = False
done_tweaking = True

# get red light kernel
datapath = '../CS148/RedLights2011_Medium/'
rl_file = 'RL-012.jpg'
rl_img = np.array(Image.open(datapath+rl_file))
init_redlightkernel = rl_img[38:54, 301:316, :]


dark_file = 'RL-334.jpg'
dark_img = np.array(Image.open(datapath+dark_file))
dark_kernel = dark_img[217:237, 307:327, :]


small_file = 'RL-269.jpg'
small_img = np.array(Image.open(datapath+small_file))
small_kernel = small_img[172:180, 325:333, :]

'''
Make predictions on the training set.
'''
preds_train = {}
for i in range(len(file_names_train)):

    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names_train[i]))

    # convert to numpy array:
    I = np.asarray(I)

    if check_dark(I, 127) == True:
        # print("using dark kernel")
        init_redlightkernel = dark_kernel

    preds_train[file_names_train[i]] = detect_red_light_mf(I, init_redlightkernel, small_kernel, count=i, plot_heatmap=False)

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds_train.json'),'w') as f:
    json.dump(preds_train,f)

if done_tweaking:
    '''
    Make predictions on the test set. 
    '''
    preds_test = {}
    for i in range(len(file_names_test)):

        # read image using PIL:
        I = Image.open(os.path.join(data_path,file_names_test[i]))

        # convert to numpy array:
        I = np.asarray(I)

        if check_dark(I, 127) == True:
            # print("using dark kernel")
            init_redlightkernel = dark_kernel

        preds_test[file_names_test[i]] = detect_red_light_mf(I, init_redlightkernel, small_kernel, count=i, plot_heatmap=False)

    # save preds (overwrites any previous predictions!)
    with open(os.path.join(preds_path,'preds_test.json'),'w') as f:
        json.dump(preds_test,f)



