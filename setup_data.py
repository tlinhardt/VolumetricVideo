import SimpleITK as sitk
import os
import numpy as np

def im_gen():
  for i in range(1,101):
    yield f'//training//patient{i:03d}//patient{i:03d}_4d.nii.gz'
  for i in range(101,151):
    yield f'//testing//patient{i:03d}//patient{i:03d}_4d.nii.gz'

def setup_data(data_path,output_path,frames=5,parsing=1):
  if not os.path.isdir(output_path):
    os.mkdir(output_path)
  n_frames = []
  for im_name in im_gen():
    #print(im_name)
    im = sitk.ReadImage(data_path+im_name)
    #print(im.GetSize())
    im = sitk.GetArrayFromImage(im[:,:,:,:])
    #print(im.shape)
    size = im.shape

    if size[1] < 8: continue

    low = [(size[1] - 8)//2,(size[2]-128)//2,(size[3]-128)//2]
    im = im[:,low[0]:low[0]+8,low[1]:low[1]+128,low[2]:low[2]+128]/255.0
    #print(im.shape)
    for j in range(0, size[0]-frames-1, parsing):
      n_frames.append(im[j:j+frames+1,:,:,:])
  np.savez(output_path+r"/data.npz", n_frames=n_frames)
  
def setup_feedback_data(data_path,output_path,frames=5,future=3,parsing=1):
  if not os.path.isdir(output_path):
    os.mkdir(output_path)
  x_train = []
  x_test = []
  y_train = []
  y_test = []
  num = 0
  for im_name in im_gen():
    im = sitk.ReadImage(data_path+im_name)
    im = sitk.GetArrayFromImage(im[:,:,:,:]).astype(np.float32)
    size = im.shape

    if size[1] < 8: continue

    low = [(size[1] - 8)//2,(size[2]-128)//2,(size[3]-128)//2]
    im = im[:,low[0]:low[0]+8,low[1]:low[1]+128,low[2]:low[2]+128]/255.0
    #print(im.shape)
    for i in range(0, size[0]-frames-future-1, parsing):
      in_frames = [ im[i+j,:,:,:] for j in range(frames) ]
      out_frames = [ im[i+j+frames,:,:,:] for j in range(future) ]
      if num < 101:
        x_train.append( in_frames )        
        y_train.append( out_frames )
      else:
        x_test.append( in_frames )
        y_test.append( out_frames )
    num += 1
  np.savez(output_path+r"/feedback_data.npz",x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)


if __name__ == "__main__":
  setup_data(r"../ImageData/",r"../TrainingData/",5,1)