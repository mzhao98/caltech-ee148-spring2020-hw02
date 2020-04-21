import os
import numpy as np
import json
from PIL import Image

def scale(im, nR, nC):
    nR0 = len(im)     # source number of rows 
    nC0 = len(im[0])  # source number of columns 
    return np.array([[ im[int(nR0 * r / nR)][int(nC0 * c / nC)]  
             for c in range(nC)] for r in range(nR)])

def scale_kernel(init_redlightkernel, scale_factor):
    # downsample redlightkernel
    #simple image scaling to (nR x nC) size
    nR = int(init_redlightkernel.shape[0]*scale_factor)
    nC = int(init_redlightkernel.shape[1]*scale_factor)
    redlightkernel = scale(init_redlightkernel, nR, nC)
    return redlightkernel

def matched_filter(img, redlightkernel):
    # dimensions of kernel
    kernel_row_size = redlightkernel.shape[0]
    kernel_col_size = redlightkernel.shape[1]
    MAX_ROWS = img.shape[0]
    MAX_COLUMNS = img.shape[1]

    # perform simple convolution
    filtered_image = np.zeros((img.shape[0], img.shape[1], 3))
    filter_h = redlightkernel # 50 x 22 x 3
    for row in range(0, MAX_ROWS, 2):
        for col in range(0, MAX_COLUMNS, 2):
            conv_value = 0
            # select corresponding window
            img_partial = img[row:min(MAX_ROWS, row+kernel_row_size), col:min(MAX_COLUMNS, col+kernel_col_size), :]
            img_partial_flat = img_partial.flatten()
            img_partial_flat_norm = (img_partial_flat - np.mean(img_partial_flat))/np.std(img_partial_flat)

            filter_img = filter_h[:min(MAX_ROWS-row, kernel_row_size), :min(MAX_COLUMNS-col, kernel_col_size), :]
            filter_img_flat = filter_img.flatten()
            filter_img_norm = (filter_img_flat - np.mean(filter_img_flat))/np.std(filter_img_flat)

            conv_value = np.dot(img_partial_flat_norm, filter_img_norm)

            filtered_image[row, col] = conv_value
    return filtered_image

def threshold_image_filter(filtered_image):
    count = 0
    rl_locations = []
    threshold = np.max(filtered_image)*(7/8)
    thresh_filtered_image = np.ones(filtered_image.shape)
    for row in range(filtered_image.shape[0]):
        for col in range(filtered_image.shape[1]):
            if filtered_image[row, col, 0] > threshold:
                count += 1
                rl_locations.append((row, col))
                thresh_filtered_image[row, col] = 0
    return count, rl_locations, thresh_filtered_image

def plot_annotated_image(img, rl_locations, redlightkernel):
    kernel_row_size = redlightkernel.shape[0]
    kernel_col_size = redlightkernel.shape[1]
    # im = thresh_filtered_image
    im = img
    # Create figure and axes
    fig,ax = plt.subplots(1)

    # Display the image
    ax.imshow(im)

    # Create a Rectangle patch
    for entry in rl_locations:
        (row, col) = entry
        rect = patches.Rectangle((col, row),kernel_col_size,kernel_row_size,linewidth=1,edgecolor='r',facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.savefig('annotated_image.png')

#k means clustering
def cluster_rl_locs(rl_locations):
    K = min(4, len(rl_locations))
    cluster_center_ids = np.random.choice(list(range(len(rl_locations))), K)
    iteration = 0
    while (1):
        
        
        cluster_groups = []
        cluster_groups_id = []
        for i in range(K):
            idx = cluster_center_ids[i]
            cluster_groups.append([rl_locations[idx]])
            cluster_groups_id.append([idx])

        for i in range(len(rl_locations)):
            min_idx = 0
            min_val = 100000
            pt_r = rl_locations[i][0]
            pt_c = rl_locations[i][1]
            for k in range(K):
                center_idx = cluster_center_ids[k]
                if center_idx == i:
                    continue

                center_r = rl_locations[center_idx][0]
                center_c = rl_locations[center_idx][1]
                dist = euclidean_dist(center_r, center_c, pt_r, pt_c)
                if dist < min_val:
                    min_val = dist
                    min_idx = k
            cluster_groups[min_idx].append((pt_r, pt_c))
            cluster_groups_id[min_idx].append(i)

        # update cluster centers
        # print(cluster_groups)
        new_cluster_center_ids = []
        for k in range(K):
            new_center_r = np.mean(np.array(cluster_groups[k])[:,0])
            new_center_c = np.mean(np.array(cluster_groups[k])[:,1])
            min_idx = 0
            min_val = 10000
            for i in range(len(cluster_groups[k])):
                dist = euclidean_dist(new_center_r, new_center_c, cluster_groups[k][i][0], cluster_groups[k][i][1])
                if dist < min_val:
                    min_val = dist
                    min_idx = cluster_groups_id[k][i]
            new_cluster_center_ids.append(min_idx)

        if list(cluster_center_ids) == list(new_cluster_center_ids) or iteration > 6:
            break
        else:
            iteration += 1
            cluster_center_ids = new_cluster_center_ids
            
    print("clustering iteration = ", iteration)
    output_locs = []
    for j in new_cluster_center_ids:
        output_locs.append(rl_locations[j])
    return output_locs

def approximate_scale(img):
    # find approximation for scale by red light size
    MAX_ROWS = img.shape[0]
    MAX_COLUMNS = img.shape[1]

    all_counts = []
    for row in range(0, MAX_ROWS):
        for col in range(0, MAX_COLUMNS):
            pixel = list(img[row, col])
            count = 0
            if pixel[0] > 230:
                if pixel[1] > 200 and pixel[2] > 200:
                    count += 1
                    col_plus = col
                    #  begin count
                    while(1):
                        col_plus += 1
                        if col_plus >= MAX_COLUMNS:
                            break

                        pixel = list(img[row, col_plus])
                        if pixel[0] > 230:
                            if pixel[1] < 180 and pixel[2] < 180:
                                count += 1
                            else:
                                break
                        else:
                            break
            if count > 3:
                all_counts.append(count)

    average_count = 7 if len(all_counts)==0 else np.mean(all_counts)
    if average_count > 13:
        average_count = 4
    return average_count/10

def euclidean_dist(x1, y1, x2, y2):
    return np.sqrt((x2-x1)**2 + (y2-y1)**2)


def detect_red_light(I, init_redlightkernel):
    '''
    This function takes a numpy array <I> and returns a list <bounding_boxes>.
    The list <bounding_boxes> should have one element for each red light in the 
    image. Each element of <bounding_boxes> should itself be a list, containing 
    four integers that specify a bounding box: the row and column index of the 
    top left corner and the row and column index of the bottom right corner (in
    that order). See the code below for an example.
    
    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''
    
    
    bounding_boxes = [] # This should be a list of lists, each of length 4. See format example below. 
    
    '''
    BEGIN YOUR CODE
    '''

    img = I
    MAX_ROWS = img.shape[0]
    MAX_COLUMNS = img.shape[1]

    scale_factor = approximate_scale(img)
    redlightkernel = scale_kernel(init_redlightkernel, scale_factor)
    filtered_image = matched_filter(img, redlightkernel)
    count, rl_locations, thresh_filtered_image = threshold_image_filter(filtered_image)
    rl_locations = cluster_rl_locs(rl_locations)

    kernel_row_size = redlightkernel.shape[0]
    kernel_col_size = redlightkernel.shape[1]

    for (row, col) in rl_locations:
        tl_row = row
        tl_col = col
        bounding_boxes.append([row, col, min(MAX_ROWS, row+kernel_row_size), min(MAX_COLUMNS, col+kernel_col_size)]) 
        

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
        
    #     bounding_boxes.append([tl_row,tl_col,br_row,br_col]) 
    
    '''
    END YOUR CODE
    '''
    
    for i in range(len(bounding_boxes)):
        assert len(bounding_boxes[i]) == 4
    
    return bounding_boxes



def check_dark(img, thrshld):
    is_light = np.mean(img) > thrshld
    return False if is_light else True







