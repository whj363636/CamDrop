import os
import json
import time
import sys


def find_checkpoint(path):
  try:
    with open(os.path.join(path, 'stats.json')) as f:
      js = json.load(f)
      start = js[0]['epoch_num']
      global_step = js[0]['global_step']
  except IOError:
    start, global_step = None, None
  return start, global_step

def find_gpus():
  os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
  memory_gpu=[int(x.split()[2]) for x in open('tmp','r').readlines()]

  count = 0
  gpu_list = ''
  for i, mem in enumerate(memory_gpu):
    if mem > 10000:
      count += 1
      gpu_list = gpu_list + str(i) + ','
    if count == 4:
      break
  gpu_list = gpu_list[:-1]
  print(memory_gpu)
  os.system('rm tmp')
  return gpu_list, count

if __name__ == '__main__':
  flag = True

  while flag:
    gpu_list, count = find_gpus()

    if count == 2:
      cmd = sys.argv[1:]
      cmd = ''.join(cmd) + ' --gpu %s' % gpu_list
      # start, global_step = find_checkpoint(main_path)
      # if start == None or global_step == None:
      #   start = 1
      #   cmd = "./%s --gpu %s --data %s -d %d --mode %s --batch %d --norm %s --keep_prob %.1f --dropblock_groups %s --ablation %s --strategy %s --blocksize %d --start %d" % ('imagenet-resnet.py', gpu_list, "/home/wangguangrun/ILSVRC2012/", 50, 'resnet', 256, 
      #           'GN', 0.9, '1,2,3,4', 'GD', 'decay', 7, start)      
      # else: 
      #   log_path = os.path.join(main_path, "model-"+str(global_step)+".data-00000-of-00001")
      #   cmd = "./%s --gpu %s --data %s -d %d --mode %s --batch %d --norm %s --keep_prob %.1f --dropblock_groups %s --ablation %s --strategy %s --blocksize %d --start %d --load %s" % ('imagenet-resnet.py', gpu_list, "/home/wangguangrun/ILSVRC2012/", 50, 'resnet', 256, 
      #           'GN', 0.9, '1,2,3,4', 'GD', 'decay', 7, start, log_path)      

      os.system(cmd)
      flag = False
    else:
      time.sleep(20)

