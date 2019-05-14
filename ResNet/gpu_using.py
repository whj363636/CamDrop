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

def find_gpus(need):
  os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
  memory_gpu=[int(x.split()[2]) for x in open('tmp','r').readlines()]

  free = 0
  gpu_list = ''
  for i, mem in enumerate(memory_gpu):
    if mem > 12000:
      free += 1
      gpu_list = gpu_list + str(i) + ','
    if free == need:
      break
  gpu_list = gpu_list[:-1]
  print(memory_gpu)
  os.system('rm tmp')
  return gpu_list, free

if __name__ == '__main__':
  flag = True

  while flag:
    need = int(sys.argv[1])
    gpu_list, free = find_gpus(need)

    if free == need:
      cmd = sys.argv[2:]
      cmd = ''.join(cmd) + ' --gpu %s' % gpu_list
      os.system(cmd)
      flag = False
    else:
      time.sleep(20)

