import numpy as np
import cv2

def tiles(batch, rows, cols, spacing_x=0, spacing_y=0, scale=1.0):
	if batch.ndim == 4:
		N, H, W, C = batch.shape
	elif batch.ndim == 3:
		N, H, W = batch.shape
		C = 1
	else:
		raise ValueError('Invalid batch shape: {}'.format(batch.shape))

	H = int(H*scale)
	W = int(W*scale)
	img = np.ones((rows*H+(rows-1)*spacing_y, cols*W+(cols-1)*spacing_x, C))
	i = 0
	for row in xrange(rows):
		for col in xrange(cols):
			start_y = row*(H+spacing_y)
			end_y = start_y + H
			start_x = col*(W+spacing_x)
			end_x = start_x + W
			if i < N:
				if C > 1:
					img[start_y:end_y,start_x:end_x,:] = cv2.resize(batch[i], (W,H))
				else:
					img[start_y:end_y,start_x:end_x,0] = cv2.resize(batch[i], (W,H))
			i += 1
	return img

def tiles4(batch, rows, cols, spacing_x=0, spacing_y=0, scale=1.0):
	if batch.ndim == 4:
		N, H, W, C = batch.shape
	assert C == 4

	H = int(H*scale)
	W = int(W*scale)
	img = np.ones((2*rows*H+(2*rows-1)*spacing_y, cols*W+(cols-1)*spacing_x, 3))
	i = 0
	for row in xrange(0,2*rows,2):
		for col in xrange(cols):
			start_y = row*(H+spacing_y)
			end_y = start_y + H
			start_x = col*(W+spacing_x)
			end_x = start_x + W
			if i < N:
				rgb = batch[i,:,:,:3]
				depth = batch[i,:,:,3:4]
				depth = np.tile(depth, (1,1,3))
				img[start_y:end_y,start_x:end_x,:] = cv2.resize(rgb, (W,H))
				img[end_y:end_y+H,start_x:end_x,:] = cv2.resize(depth, (W,H))
			i += 1
	return img