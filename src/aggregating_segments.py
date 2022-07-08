inputs_part_save = []
output_part_save = []
inputs_part_avg = [] 
output_part_avg = []
name_num=1
Nz=64
Ns=64
vt=0
for fn in range(15):
 
  csib_batch_iterator = BatchIterator(
      batch_size=1,
      keys=['images', 'labels'],
      data=csib_dataset[vt:vt+4],
      global_transforms=global_transforms_test,
      shuffle=False
  )
  vt=vt+4


  for data in csib_batch_iterator:   
      inputs_all = torch.from_numpy(data['images'])
      depth =  np.squeeze(np.sum(np.sum(inputs_all.numpy(),axis=3),axis=2))

      outputs = np.squeeze(np.transpose(np.zeros(inputs_all.shape),[2,3,4,0,1]))
      inputs = np.squeeze(np.transpose(np.zeros(inputs_all.shape),[2,3,4,0,1]))

      part_num = (len(depth)//Ns)
      avg_num = part_num-1
      st=0
      for i in range(part_num):
          print(i)
          inputs_part = inputs_all[ :, :, :, :, st:st+Nz]
          inputs_part = Variable(inputs_part)
          if torch.cuda.is_available:
              inputs_part = inputs_part.cuda()

          output_part = net_seg(inputs_part)
          output_part = torch.nn.Sigmoid()(output_part)
          inputs_part_save.append(inputs_part.detach())
          output_part_save.append(output_part.detach())
          st+=Ns

      st=0
      for i in range(part_num):
          print(i)
          tmp1=np.squeeze(np.transpose(output_part_save[i].cpu().numpy(),[2,3,4,0,1]))
          tmp2=np.squeeze(np.transpose(inputs_part_save[i].cpu().numpy(),[2,3,4,0,1]))
          for i in range(4):
                tmp=tmp1[:,:,:,i]
                tmp=np.reshape(tmp,(192,192,64))
                outputs[ :, :, st:st+Nz] += tmp[:,:,:]
          inputs[ :, :, st:st+Nz] = inputs[ :, :, st:st+Nz]+tmp2[:,:,:]
          st+=Ns


      new_image = nib.Nifti1Image(inputs/16, affine=np.eye(4))
      new_image.header['pixdim']=[1., image_resolution[0],image_resolution[1],image_resolution[2],1.,1.,1.,1.]
      nib.save(new_image,os.path.join(test_all,'test'+str(name_num)+'image_.nii.gz'))
      new_image = nib.Nifti1Image(outputs/16, affine=np.eye(4))
      new_image.header['pixdim']=[1., image_resolution[0],image_resolution[1],image_resolution[2],1.,1.,1.,1.]
      nib.save(new_image,os.path.join(test_all,'test'+str(name_num)+'label_.nii.gz'))
      inputs_part_save = []
      output_part_save = []
      inputs_part_avg = [] 
      output_part_avg = []
      name_num+=1
