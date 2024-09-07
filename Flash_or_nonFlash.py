import cv2
import numpy as np

def split(img):  #process each color separate
  r = img[:,:,0]
  g = img[:,:,1]
  b = img[:,:,2]
  return r,g,b

def process(img):  #pixel values in range,splits it
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img = img.astype('double')/255
  return split(img)

def thresh(mask, f):  #creates binary mask and isolates areas with significant changes
  flag = np.zeros((mask.shape), np.uint8)
  flag[(mask > -0.2) & (mask < -0.05)] = 1
  flag[(mask > 0.65) & (mask < 0.7)] = 1
  flag[f > 0.95*(np.max(f) - np.min(f))] = 1
  return flag

def dia(radius):
  diameter = 2 * radius + 1
  return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))

def morph(flag):    #performs erosion, flood filling, dilation, and additional erosion to clean up the mask
  d1 = dia(2)       #removes noise, fills gaps, and improves the accuracy of the mask
  d2 = dia(6)
  d3 = dia(4)

  flag = cv2.erode(flag, d1, iterations = 1)  #removes small noise elements
  mask = np.zeros((flag.shape[0]+2, flag.shape[1]+2), np.uint8)  
  cv2.floodFill(flag, mask, (0,0), 1)  #fills gaps
  mask = 1 - mask
  mask = cv2.dilate(mask, d2)  #expands mask
  mask = cv2.erode(mask, d3)  #smooth boundaries
  morphed_mask = mask.astype('double')

  return morphed_mask

def filter(mask):  #smooths mask
  kernel = np.array([
  [0.1070,    0.1131,    0.1070],
  [0.1131,    0.1196,    0.1131],
  [0.1070,    0.1131,    0.1070]
  ])
  mask = cv2.filter2D(mask, -1, kernel)  #applies convolutional filter to mask
  return mask

def create_mask(f,nof):
  mask = f - nof   #diff b f and nf to know where does f has impact
  flash_mask = thresh(mask, f)    #diff calc, binary mask
  morph_mask = morph(flash_mask)    #improves mask
  filter_mask = filter(morph_mask)  #smooths and add a layer
  return filter_mask[:-2,:-2]   #final mask, cropped to remove padding

def gaussTo1(r,g,b):  #rgb to grayscale using weighted sum
  return 0.299*r + 0.587*g + 0.114*b

def gaussian_kernel(k, sig):
  x = np.linspace(-(k//2), (k//2), k)
  gauss_x, gauss_y = np.meshgrid(x, x)
  kernel = np.exp(-1*(gauss_x**2 + gauss_y**2)/(2*(sig**2)))
  return kernel

def parameters(h):
  if(h==636):
    return 7, 4, 1.5
  elif(h==706):
    return 13, 9, 2
  elif(h==563):
    return 7, 6, 7
  else:
    return 7, 8, 1.5

def bilateral(flash, nonflash):    #apply bilateral filter to each color chanel to separate into
  k, sigma_s, sigma_r = parameters(flash.shape[0])    #joint, base and output base
  gauss_mask = gaussian_kernel(k, sigma_s)

  bias = (k//2)

  flash_pad = np.lib.pad(flash, (bias, bias), 'edge')
  non_flash_pad = np.lib.pad(nonflash, (bias, bias), 'edge')

  h, w = flash.shape
  non_flash_joint = np.zeros((h, w))
  non_flash_base = np.zeros((h, w))
  output_base = np.zeros((h, w))

  for i in range(bias, h+bias):
    for j in range(bias, w+bias):

      non_flash_mask = non_flash_pad[i-bias:i+bias+1, j-bias:j+bias+1]
      flash_mask = flash_pad[i-bias:i+bias+1, j-bias:j+bias+1]

      flash_diffmask = flash_mask - flash_pad[i, j]
      non_flash_diff = non_flash_mask - non_flash_pad[i, j]

      BF_flash_mask = np.exp(-1*((flash_diffmask/sigma_r)**2)/(2*(sigma_r**2)))
      BF_non_flash_mask = np.exp(-1*((non_flash_diff/sigma_r)**2)/(2*(sigma_r**2)))

      filt_mask_flash = BF_flash_mask*gauss_mask
      norm_term_flash = np.sum(filt_mask_flash)

      filt_mask_non_flash = BF_non_flash_mask*gauss_mask
      norm_term_non_flash = np.sum(filt_mask_non_flash)

      non_flash_joint_mask = (non_flash_mask*filt_mask_flash)/norm_term_flash
      non_flash_base_mask = (non_flash_mask*filt_mask_non_flash)/norm_term_non_flash
      output_base_mask = (flash_mask*filt_mask_flash)/norm_term_flash

      non_flash_joint[i-bias, j-bias] = np.sum(non_flash_joint_mask)
      non_flash_base[i-bias, j-bias] = np.sum(non_flash_base_mask)
      output_base[i-bias, j-bias] = np.sum(output_base_mask)

  return [non_flash_joint, non_flash_base, output_base]

def get_detail(f,base): #gets detail by dividing flash/base
  return (f + 0.02)/(base + 0.02)

def cvTo(image):
  image_normalized = cv2.normalize(image, None, 0.0, 255.0, cv2.NORM_MINMAX)
  image_int = cv2.convertScaleAbs(image_normalized)
  img = cv2.cvtColor(image_int,cv2.COLOR_RGB2BGR)
  return img

def solution(image_path_a, image_path_b):

  fimg = cv2.imread(image_path_b)
  nfimg = cv2.imread(image_path_a)

  fr,fg,fb = process(fimg)
  nfr,nfg,nfb = process(nfimg)

  f = gaussTo1(fr,fg,fb)
  nof = gaussTo1(nfr,nfg,nfb)
  shadow_mask = create_mask(f,nof)
  
#bilateral filtering to each color channel
  [non_flash_joint_r, non_flash_base_r, output_base_r] = bilateral(fr, nfr)
  [non_flash_joint_g, non_flash_base_g, output_base_g] = bilateral(fg, nfg)
  [non_flash_joint_b, non_flash_base_b, output_base_b] = bilateral(fb, nfb)

  output_detail_r = get_detail(fr,output_base_r)
  output_detail_g = get_detail(fg,output_base_g)
  output_detail_b = get_detail(fb,output_base_b)
  output_detail = np.dstack((output_detail_r, output_detail_g, output_detail_b))
  
#Combines the enhanced detail from the flash image with the no-flash image based on the shadow mask
  non_flash_joint = np.dstack((non_flash_joint_r, non_flash_joint_g, non_flash_joint_b))
  non_flash_base = np.dstack((non_flash_base_r, non_flash_base_g, non_flash_base_b))

  output_fin = (np.dstack(((1-shadow_mask), (1-shadow_mask), (1-shadow_mask)))*(non_flash_joint*output_detail) + np.dstack((shadow_mask, shadow_mask, shadow_mask))*(non_flash_base))

  output_fin[output_fin>1] = 1  #ensures range is [0,1] and converts to 8 bit 
  output_detail[output_detail>1] = 1

  img = cvTo(output_fin)

  return img
