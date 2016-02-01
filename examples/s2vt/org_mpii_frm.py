#import caffe
import os
import glob
base_dir = '/media/researchshare/linjie/data/MPII/'
img_dir ='/media/researchshare/linjie/data/MPII/jpgAllFrames/'
sav_dir ='/media/researchshare/linjie/data/MPII/googlenet/'
im_inter = 5 #according to the s2s paper
stage='test'
fds = []
with open(base_dir+'dataSplit.txt','r') as splitfile:
	for line in splitfile:
		line=line[0:-2]
		content = line.split('\t')
		fd = content[0]
		s = content[1]
		print s
		if s==stage:
			fds.append(fd)

print len(fds)
#load the model
fout = open('mpii_'+stage+'_list_frm.txt','w')
#fds = [fd for fd in os.listdir(img_dir) if os.path.isdir(img_dir+fd)]
for fd in fds:
	sub_fds = [sub_f for sub_f in os.listdir(img_dir+fd) if os.path.isdir(img_dir+fd+'/'+sub_f)]
	for sub_fd in sub_fds:
		sub_path = img_dir + fd + '/' + sub_fd
		im_list = os.listdir(sub_path)
		im_n = len(im_list)
		for i in xrange(1,im_n+1,im_inter):
			fout.write('%s/%04d.jpg 0\n' % (sub_path,i))

fout.close()

